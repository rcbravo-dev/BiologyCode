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
    # print(error)
    cwd = os.getcwd()
    script_dir = cwd + '/main' 

# Add the script directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))

from lib.codebase import GeneticPool
from lib.configurator import configurations as cfg
from lib.utilities import valid_log_level

# Run once in the main script
logging.config.dictConfig(cfg.logging_config)

# Include in each module:
LOG = logging.getLogger('SERVER')


class GeneticServer:
    def __init__(self, 
                 host: str, 
                 port: int, 
                 pool_size: int = 512, 
                 tape_length: int = 64, 
                 epochs: int = 100,
                 mutation_rate: float = 0.05,
                 experiment_name: str = 'run_001', 
                 track_generations: bool = True,
                 first_run: bool = True,
                 verbose: bool = True,
                 ) -> None:
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.tape_length = tape_length
        self.epochs = epochs
        self.epochs_remaining = self.epochs
        self.mutation_rate = mutation_rate
        self.path = f'{cfg.data_path}/server/{experiment_name}/'
        self.track_generations = track_generations
        self.verbose = verbose
        self.generator = None
        self.first_run = first_run
        self.time_out = cfg.handler_timeout
        self.__post_init__()

    def __post_init__(self):
        '''Post initialization method.'''
        self.rng = cfg.rng
        self.gp = GeneticPool(self.pool_size, self.tape_length, server_path=self.path)
        self.tx_queue = asyncio.Queue()
        self.SENTINEL = object()
        self.server_ready_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.genes = {}
        
    async def initialize_server(self):
        '''Initialize the server.'''
        load_most_recent = True

        if self.first_run and self.track_generations:
            pass
        elif self.first_run and not self.track_generations:
            # After clearing the server path, will load the most recent pool,
            # which will raise a FileNotFoundError (Because any files are 
            # removed with clear_server_path is called), then create and save a new pool.
            self.gp.clear_server_path()
        elif not self.first_run:
            self.epochs_remaining = self.epochs

            if hasattr(self.gp, 'pool'):
                load_most_recent = False
        
        try:
            if load_most_recent:
                # Get the most recent
                self.gp.load_most_recent()
        except FileNotFoundError:
            # If not found create and save
            self.gp.create()
            await self._save_pool()
        finally:
            self.first_run = False
            await self._producer_task()

    async def run_server(self):
        await self.initialize_server()
        
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

    async def handle_client(self, request):
        """Handler for client requests to get byte arrays."""
        try:
            # Wait until the server is ready, but with a timeout
            await asyncio.wait_for(self.server_ready_event.wait(), timeout=self.time_out)

            # Get data from the queue
            byte_array = await self.tx_queue.get()
            
            if byte_array is self.SENTINEL:
                await self._on_epoch_completion()
                return await self.handle_client(request)
            
            elif byte_array == 'STOP':
                await self._on_epoch_completion()
                await self._save_pool()
                raise asyncio.TimeoutError('STOP')
                
        except asyncio.TimeoutError:
            if self.epochs_remaining:
                LOG.debug(f'Server temporarily unavailable. epochs={self.epochs_remaining}, ready={self.server_ready_event.is_set()}, {self.tx_queue.qsize()} items in queue.')
                return web.Response(status=204, text="Server temporarily unavailable.")
            else:
                LOG.debug(f'All epochs have been served.')
                return web.Response(status=408, text='STOP')
        except Exception as e:
            LOG.exception(f"Error in handle_client: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return web.Response(status=202, body=byte_array) 
        
    async def handle_reset(self, request):
        try:
            await self.initialize_server()
            await asyncio.sleep(0)
        except Exception as e:
            LOG.exception(f"Error in handle_reset: {e}", exc_info=True)
        else:
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
            LOG.exception(f"Error in handle_returned_data: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return await self.handle_client(request) 

    async def _producer_task(self):
        """Background task to populate the queue with byte arrays."""
        if self.epochs_remaining > 0:
            # Indicate that the server is not ready while repopulating
            self.server_ready_event.clear()

            # Initialize the generator if it's not set
            if self.generator is None:
                self.generator = self._byte_array_generator()
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

            LOG.debug(f'Producer task complete. Epochs remaining: {self.epochs_remaining}')

    async def _byte_array_generator(self):
        # Async generator that yields 128-byte arrays
        tapes = self.rng.permutation(self.pool_size).astype(np.uint16)
        
        for i, j in zip(tapes[0::2], tapes[1::2]):
            header = np.array([i, j], dtype=np.uint16).tobytes()
            # Pool should be a long byte string
            tape_i = self.gp[i]
            tape_j = self.gp[j]
            yield header + tape_i + tape_j

    async def _on_epoch_completion(self):
        '''Actions to take when an epoch is completed.'''
        # Indicate that the server is not ready while repopulating
        self.server_ready_event.clear()
        # Find dominate gene from the pool before mutating
        await self._find_dominate_gene_from_pool()
        # Mutate the pool
        await self._mutate_pool()
        # Repopulate the queue with mutated tapes
        await self._producer_task()
        # Sleep to allow other tasks to run
        await asyncio.sleep(0)

    async def _mutate_pool(self):
        if self.mutation_rate:
            actual_mutation_rate = self.gp.mutation(rate=self.mutation_rate)
        
            LOG.debug(f'Epoch {self.epochs_remaining} mutated pool with rate: {actual_mutation_rate}')
        else:
            LOG.debug(f'Epoch {self.epochs_remaining} pool not mutated.')

    async def _find_dominate_gene_from_pool(self) -> tuple[str, float]:
        gene = self.gp.find_dominate_gene()
        gene_str, fitness = self.gp.determine_gene_fitness(gene)

        if gene_str not in self.genes:
            self.genes[gene_str] = fitness
        
        if self.verbose:
            print(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs: {self.epochs_remaining}')
        
        LOG.debug(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs remaining: {self.epochs_remaining}')

    async def _save_pool(self):
        '''Save the pool.'''
        # Save the pool before stopping - Doing it here
        # so is saved only once.
        if self.track_generations:
            self.gp.save(overwrite=False)
        else:
            self.gp.save(overwrite=True, safe=False)
        
        # Log compression ratio
        LOG.info(f'Pool saved after {self.epochs - self.epochs_remaining} epochs with a compression ratio of: {round(self.gp.compression_ratio, 4)}.')


# python3 -m server -hs 'localhost' -p 8080 -e 100 -ps 512 -t 64 -m 0.03 -en 'run_001' -v --track_gen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server the GeneticCode.')
    parser.add_argument('-hs', '--host', default=cfg.api_host, type=str, help='ip address or "localhost" of server.')  
    parser.add_argument('-p', '--port', default=cfg.api_port, type=int, help='Port of server.') 
    parser.add_argument('-e', '--epoch', default=cfg.epochs, type=int, help='Number of times to run all tapes.')
    parser.add_argument('-ps', '--pool_size', default=cfg.pool_size, type=int, help='Number of tapes in the pool.')
    parser.add_argument('-t', '--tape_length', default=cfg.tape_length, type=int, help='Number of bytes in the tape.')
    parser.add_argument('-m', '--mutation_rate', default=cfg.mutation_rate, type=float, help='Rate of mutation.')
    parser.add_argument('-en', '--experiment_name', default='run_001', type=str, help='Name of the genetic pool.')
    parser.add_argument('--track_gen', action='store_true', help='Initialize the server and run forever.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity.')
    parser.add_argument('--log-level', type=valid_log_level, default=cfg.log_level, 
                        help=f'Set the logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is {cfg.log_level}.')
    args = parser.parse_args()

    # Logging
    logging.config.dictConfig(cfg.logging_config)
    LOG.setLevel(args.log_level)
    LOG.debug(f"Server logging is configured: {LOG}.")

    try:
        s = GeneticServer(
            args.host,
            args.port,
            pool_size = args.pool_size,
            tape_length = args.tape_length,
            mutation_rate = args.mutation_rate,
            epochs = args.epoch,
            experiment_name = args.experiment_name,
            track_generations = args.track_gen,
            verbose = args.verbose,
        )
        asyncio.run(s.run_server())
    except KeyboardInterrupt:
        if s.epochs_remaining:
            # Save the pool before stopping if there are epochs remaining
            # signifying that there was a stoppage and the server did not
            # have a chance to save the pool.
            asyncio.run(s._save_pool())

        LOG.info('Server stopped by user.')
        print('Server stopped by user.', end='\n\n')
    except Exception as e:
        LOG.exception(f"Error in main: {e}", exc_info=True)

        