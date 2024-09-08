'''
This code implements the server side of a Zero Knowledge Protocol (ZKP) 
Application. ZKP is a method for two parties to prove to each other that 
they have full knowledge of a secret without revealing the secret itself.

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

# from lib.zka_networking import ThreadedTCPRequestHandler
# from lib.new_qrcode import RegistrationExchange
# from lib.zka_crypto import generate_rsa_key_pair
from lib.codebase import GeneticPool, find_dominate_gene, determine_gene_fitness
# from lib.new_server_api import GenesisVerify, ChallengeVerify, MasterSecretVerify
# from lib.new_database import AsyncDatabase
from lib.configurator import configurations as cfg

import logging
import logging.config
# Run once in the main script
logging.config.dictConfig(cfg.logging_config)
# Include in each module:
LOG = logging.getLogger('SERVER')
LOG.debug("Server logging is configured.")

LOG.setLevel(logging.INFO)
# Shared state
tx_queue = asyncio.Queue()
rx_queue = asyncio.Queue()
server_ready_event = asyncio.Event()
SENTINEL = object()  # Special object to signal exhaustion
lock = asyncio.Lock()
generator = None
ready = True
genes = {}

# Params
epochs_remaining = 200  # Example value for the number of epochs to run
pool_size = 512
tape_length = 64
path = f'{cfg.data_path}/server/run_001/'

# Resources
rng = np.random.default_rng()
gp = GeneticPool(pool_size, tape_length, filename=path)
gp.create()
# gp.save(overwrite=True) 
gp.load(filename=f'{path}genetic_pool_2')
gp.pool = bytearray(gp.pool)


async def byte_array_generator():
    # Async generator that yields 128-byte arrays
    tapes = rng.permutation(pool_size).astype(np.uint16)
    
    for i, j in zip(tapes[0::2], tapes[1::2]):
        header = np.array([i, j], dtype=np.uint16).tobytes()
        # Pool should be a long byte string
        tape_i = gp[i]
        tape_j = gp[j]
        yield header + tape_i + tape_j

async def find_dominate_gene_from_pool() -> tuple[str, float]:
    global genes, epochs_remaining

    pool = np.frombuffer(gp.pool, dtype=np.uint8)

    gene = find_dominate_gene(pool, pool_size, tape_length)
    gene_str, fitness = determine_gene_fitness(gene)

    if gene_str not in genes:
        genes[gene_str] = fitness
    
    print(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs remaining: {epochs_remaining}')
    
    # LOG.info(f'Dominant Gene: {gene_str}, fitness: {fitness}, epochs remaining: {epochs_remaining}')

async def producer_task():
    """Background task to populate the queue with byte arrays."""
    global generator, epochs_remaining

    if epochs_remaining > 0:
        # Indicate that the server is not ready while repopulating
        server_ready_event.clear()

        # Initialize the generator if it's not set
        if generator is None:
            generator = byte_array_generator()
            epochs_remaining -= 1

        # Repopulate the queue
        try:
            async for byte_array in generator:
                await tx_queue.put(byte_array)
            if epochs_remaining > 0:
                # Signals the end of the epoch
                await tx_queue.put(SENTINEL)  
            else:
                # Signals the end of all epochs
                await tx_queue.put('STOP')
        except StopAsyncIteration:
            pass
        finally:
            generator = None  # Reset generator when exhausted

        # Signal that the server is ready
        server_ready_event.set()
        
        # Wait for a condition before starting the next epoch
        await asyncio.sleep(0)

        LOG.info(f'Epochs remaining: {epochs_remaining}')

async def handle_client(request):
    global ready
    
    """Handler for client requests to get byte arrays."""
    try:
        # Wait until the server is ready, but with a timeout
        await asyncio.wait_for(server_ready_event.wait(), timeout=5)

        # Get data from the queue
        byte_array = await tx_queue.get()
        
        if byte_array is SENTINEL:
            # Indicate that the server is not ready while repopulating
            server_ready_event.clear()
            # Repopulate the queue in the background
            asyncio.create_task(producer_task())
            await find_dominate_gene_from_pool()
            await asyncio.sleep(0)
            return await handle_client(request)
        elif byte_array == 'STOP':
            ready = False
            # Indicate that the server is not ready while repopulating
            server_ready_event.clear()
            # Save the pool before stopping - Doing it here
            # so is saved only once.
            gp.save(overwrite=False)
            raise asyncio.TimeoutError('STOP')
            
    except asyncio.TimeoutError:
        if ready:
            LOG.debug(f'Server temporarily unavailable. epochs={epochs_remaining}, ready={server_ready_event.is_set()}, {tx_queue.qsize()} items in queue.')
            return web.Response(status=204, text="Server temporarily unavailable.")
        else:
            LOG.debug(f'All epochs have been served.')
            return web.Response(status=408, text='STOP')
    except Exception as e:
        LOG.error(f"Error in handle_client: {e}", exc_info=True)
        return web.Response(status=500, text='Server error')
    else:
        return web.Response(status=202, body=byte_array, ) 

async def handle_returned_data(request):
    """Handler for client requests to return processed byte arrays."""
    try:
        data = await request.content.read()
        
        i, j = np.frombuffer(data[:4], dtype=np.uint16)
        async with lock:
            gp[i] = data[4:tape_length+4]
            gp[j] = data[tape_length+4:]

        LOG.debug(f'Checked in tapes {i} and {j}')
    except Exception as e:
        LOG.error(f"Error in handle_returned_data: {e}", exc_info=True)
        return web.Response(status=500, text='Server error')
    else:
        return await handle_client(request) 

async def main():
    # Start the producer task to populate the queue
    # asyncio.create_task(producer_task())
    await producer_task()

    # Set up the web server
    app = web.Application()
    app.router.add_get('/get', handle_client)
    app.router.add_post('/post', handle_returned_data)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    print("Server running on http://localhost:8080")

    # Keep the server running
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(main())