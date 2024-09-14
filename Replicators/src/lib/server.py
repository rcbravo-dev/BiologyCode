'''
This code implements the server side of a Replicators API that serves a 
computational analog to RNA.

Copyright (C) 2024  RC Bravo Consuling Inc., https://github.com/rcbravo-dev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import os
import sys
import argparse
from aiohttp import web
import asyncio
import numpy as np
import logging
import logging.config

# Get the directory of the current script
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
except NameError as error:
    import os
    print(error)
    cwd = os.getcwd()
    script_dir = cwd + '/main' 

# Add the script directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))

from lib.codebase import GeneticPool
from lib.configurator import configurations as cfg

# Run once in the main script
logging.config.dictConfig(cfg.logging_config)
# Include in each module:
LOG = logging.getLogger('SERVER')
LOG.debug("Server logging is configured.")
LOG.setLevel(logging.INFO)


class GeneticServer:
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 8080, 
                 pool_size: int = 512, 
                 tape_length: int = 64, 
                 epochs: int = 100,
                 mutation_rate: float = 0.05,
                 experiment_name: str = 'run_001', 
                 track_generations: bool = True,
                 ) -> None:
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.tape_length = tape_length
        self.epochs = epochs
        self.epochs_remaining = epochs
        self.mutation_rate = mutation_rate
        self.path = f'{cfg.data_path}/server/{experiment_name}/'
        self.rng = np.random.default_rng()
        self.gp = GeneticPool(pool_size, tape_length, filename=self.path)
        self.genes = {}
        self.tx_queue = asyncio.Queue()
        self.SENTINEL = object()
        self.server_ready_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.generator = None
        self.first_run = True
        self.track_generations = track_generations

    def initialize_server(self):
        if self.first_run:
            self.gp.create()
            self.gp.save(overwrite=True)
            self.first_run = False
        else:
            self.gp.load_most_recent()

    async def run_server(self):
        self.initialize_server()

        await self.producer_task()

        # Set up the web server
        app = web.Application()
        app.router.add_get('/get', self.handle_client)
        app.router.add_get('/reset', self.handle_reset)
        app.router.add_post('/post', self.handle_returned_data)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"Server running on http://{self.host}:{self.port}")

        # Keep the server running
        while True:
            await asyncio.sleep(3600)

    async def byte_array_generator(self):
        # Async generator that yields 128-byte arrays
        tapes = self.rng.permutation(self.pool_size).astype(np.uint16)
        
        for i, j in zip(tapes[0::2], tapes[1::2]):
            header = np.array([i, j], dtype=np.uint16).tobytes()
            # Pool should be a long byte string
            tape_i = self.gp[i]
            tape_j = self.gp[j]
            yield header + tape_i + tape_j

    async def producer_task(self):
        """Background task to populate the queue with byte arrays."""
        if self.epochs_remaining > 0:
            # Indicate that the server is not ready while repopulating
            self.server_ready_event.clear()

            # Initialize the generator if it's not set
            if self.generator is None:
                self.generator = self.byte_array_generator()
                self.epochs_remaining -= 1

            # Repopulate the queue
            try:
                async for byte_array in self.generator:
                    await self.tx_queue.put(byte_array)
                
            except StopAsyncIteration:
                pass
            finally:
                if self.epochs_remaining > 0:
                    # Signals the end of the epoch
                    await self.tx_queue.put(self.SENTINEL)  
                else:
                    # Signals the end of all epochs
                    await self.tx_queue.put('STOP')  

                # Reset generator when exhausted
                self.generator = None  

            # Signal that the server is ready
            self.server_ready_event.set()
            
            # Wait for a condition before starting the next epoch
            await asyncio.sleep(0)

            LOG.debug(f'Epochs remaining: {self.epochs_remaining}')

    async def handle_client(self, request):
        """Handler for client requests to get byte arrays."""
        try:
            # Wait until the server is ready, but with a timeout
            await asyncio.wait_for(self.server_ready_event.wait(), timeout=5)

            # Get data from the queue
            byte_array = await self.tx_queue.get()
            
            if byte_array is self.SENTINEL:
                # Indicate that the server is not ready while repopulating
                self.server_ready_event.clear()
                # Find dominate gene from the pool before mutating
                await self.find_dominate_gene_from_pool()
                # Mutate the pool - must be done prior to repopulating the queue
                await self.mutate_pool()
                # Repopulate the queue with mutated tapes
                await self.producer_task()
                await asyncio.sleep(0)
                return await self.handle_client(request)
            
            elif byte_array == 'STOP':
                # Indicate that the server is not ready while repopulating
                self.server_ready_event.clear()
                # Find the dominate gene of epoch 0
                await self.find_dominate_gene_from_pool()
                # Save the pool before stopping - Doing it here
                # so is saved only once.
                if self.track_generations:
                    self.gp.save(overwrite=False)
                else:
                    self.gp.save(overwrite=True)
                # Log compression ratio
                LOG.info(f'Compression ratio: {round(self.gp.compression_ratio, 4)} epoch: {self.epochs_remaining}')
                raise asyncio.TimeoutError('STOP')
                
        except asyncio.TimeoutError:
            if self.epochs_remaining:
                LOG.debug(f'Server temporarily unavailable. epochs={self.epochs_remaining}, ready={self.server_ready_event.is_set()}, {self.tx_queue.qsize()} items in queue.')
                return web.Response(status=204, text="Server temporarily unavailable.")
            else:
                LOG.debug(f'All epochs have been served.')
                return web.Response(status=408, text='STOP')
        except Exception as e:
            LOG.error(f"Error in handle_client: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return web.Response(status=202, body=byte_array, ) 
        
    async def handle_reset(self, request):
        self.epochs_remaining = self.epochs
        await self.mutate_pool()
        await self.producer_task()
        await asyncio.sleep(0)
        return await self.handle_client(request)

    async def handle_returned_data(self, request):
        """Handler for client requests to return processed byte arrays."""
        try:
            data = await request.content.read()
            tape_length = self.tape_length
            
            i, j = np.frombuffer(data[:4], dtype=np.uint16)
            async with self.lock:
                self.gp[i] = data[4:tape_length+4]
                self.gp[j] = data[tape_length+4:]

            LOG.debug(f'Checked in tapes {i} and {j}')
        except Exception as e:
            LOG.error(f"Error in handle_returned_data: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return await self.handle_client(request) 
    
    async def mutate_pool(self):
        actual_mutation_rate = self.gp.mutation(rate=self.mutation_rate)
        LOG.debug(f'Epoch {self.epochs_remaining} mutated pool with rate: {actual_mutation_rate}')

    async def find_dominate_gene_from_pool(self) -> tuple[str, float]:
        gene = self.gp.find_dominate_gene()
        gene_str, fitness = self.gp.determine_gene_fitness(gene)

        if gene_str not in self.genes:
            self.genes[gene_str] = fitness
        
        print(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs: {self.epochs_remaining}')
        
        # LOG.info(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs remaining: {epochs_remaining}')

if __name__ == '__main__':
    # python3 -m server -hs 'localhost' -p 8080 -e 100 -ps 512 -t 64 -m 0.05 -en 'run_001' --track_gen

    parser = argparse.ArgumentParser(description='Server the GeneticCode.')
    parser.add_argument('-hs', '--host', default='localhost', help='ip address or "localhost" of server.')  
    parser.add_argument('-p', '--port', default=8080, help='Port of server.') 
    parser.add_argument('-e', '--epoch', default=10, help='Number of times to run all tapes.')
    parser.add_argument('-ps', '--pool_size', default=100, help='Number of tapes in the pool.')
    parser.add_argument('-t', '--tape_length', default=64, help='Number of bytes in the tape.')
    parser.add_argument('-m', '--mutation_rate', default=0.05, help='Rate of mutation.')
    parser.add_argument('-en', '--experiment_name', default='run_001', help='Name of the genetic pool.')
    parser.add_argument('--track_gen', action='store_true', help='Initialize the server and run forever.')
    args = parser.parse_args()

    if args.track_gen:
        track_generations = True
    else:
        track_generations = False

    try:
        s = GeneticServer(
            host = args.host,
            port = int(args.port),
            pool_size = int(args.pool_size),
            tape_length = int(args.tape_length),
            mutation_rate = float(args.mutation_rate),
            epochs = int(args.epoch),
            experiment_name = args.experiment_name,
            track_generations = track_generations,
        )
        asyncio.run(s.run_server())
    except KeyboardInterrupt:
        # Save the pool before stopping
        s.gp.save(overwrite=False)

        LOG.info('Server stopped by user.')
        print('Server stopped by user.', end='\n\n')

        