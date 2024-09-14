'''
This code implements the client side of a Replicators API that combines 
computational analogs of RNA via replication, mutation, and selection. 

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
import logging
import logging.config
import numpy as np
import aiohttp
import asyncio

# Get the directory of the current script
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    cwd = os.getcwd()
    script_dir = cwd + '/main'

# Add the script directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))

from lib.codebase import GeneticExchangeClient
from lib.configurator import configurations as cfg

# Run once in the main script
logging.config.dictConfig(cfg.logging_config)

# Include in each module:
LOG = logging.getLogger('CLIENT')
LOG.debug("Client logging is configured.")
LOG.setLevel(logging.INFO)


class Client:
    def __init__(self, host: str, port: int, client_name: str) -> None:
        self.url = f'http://{host}:{port}'
        self.client_name = client_name
        self.data_path = cfg.data_path
        self.client_path = f'{self.data_path}/clients/{self.client_name}'
        if not os.path.exists(self.client_path):
            os.makedirs(self.client_path)
        self.header = np.array([10], dtype=np.uint8)
        self.gce = GeneticExchangeClient()
        self.sleep_count = 10
        self.sleep_time = 0.1

    async def worker(self) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                
                response = await session.get(f'{self.url}/get', data='NEW WORKER')
                
                while response.status == 202:
                    data = await response.read()

                    header = data[:4]
                    tape = data[4:]
                    
                    tape = self.gce.run(tape)
                    response = await session.post(f'{self.url}/post', data=header + tape)
                    LOG.debug(f"Message - {tape[:10]}, response - {response.status} {self.client_name}")

                await self.status_handler(response)
        except aiohttp.client_exceptions.ServerDisconnectedError as e:
            print(f'Connection to server lost: {e}')
            await self.work_waiter()
        except Exception as e:
            LOG.error(f'worker error={e}', exc_info=True)
            raise(e)
        
    async def reset(self) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                
                response = await session.get(f'{self.url}/reset', data='RESET SERVER')
                
                while response.status == 202:
                    data = await response.read()

                    header = data[:4]
                    tape = data[4:]
                    
                    tape = self.gce.run(tape)
                    response = await session.post(f'{self.url}/post', data=header + tape)
                    LOG.debug(f"Message - {tape[:10]}, response - {response.status} {self.client_name}")

                await self.status_handler(response)
        except aiohttp.client_exceptions.ServerDisconnectedError as e:
            print(f'Connection to server lost: {e}')
            await self.work_waiter()
        except Exception as e:
            LOG.error(f'worker error={e}', exc_info=True)
            raise(e)

    async def status_handler(self, response: aiohttp.ClientResponse) -> None:
        print(f'Client: {self.client_name}')
        
        msg = await response.text()

        if response.status == 204:
            print(f'Server temporarily unavailable.')
            print()
            await self.work_waiter()
        elif response.status == 400:
            print(f'ValueError. Status: {response.status} {msg}')
        elif response.status == 408:
            print(f'All tapes have been processed. Status: {response.status} {msg}')
            print(f'Runs: {len(self.gce.times)}')
            print(f'Instruction Counts: {np.sum(self.gce.instruction_counts)}')
            print(f'Total Time: {np.sum(self.gce.times)}')
            print(f'Average Time: {np.mean(self.gce.times)}')
            print(f'Loop Errors: {self.gce.loop_errors}')
            print(f'Max Read Errors: {self.gce.exaustion_errors}')
            print()
        else:
            print(f'Worker Error Status: {response.status} {msg}')

    async def work_waiter(self) -> None:
        await asyncio.sleep(self.sleep_time)
        self.sleep_count -= 1
        self.sleep_time += 0.1
        if self.sleep_count == 0:
            raise TimeoutError('Server is not responding after 10 tries.')
        else:
            await self.worker()


# Example usage
# python3 -m client --work -hs 'localhost' -p 8080 -c alice

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a message in bytes format to a server.')
    
    parser.add_argument('--work', action='store_true', help='Start worker.')
    parser.add_argument('--reset', action='store_true', help='Reset Server.')
    parser.add_argument('-hs', '--host', default='localhost', help='ip address or "localhost" of server.')  
    parser.add_argument('-p', '--port', default=9999, help='Port of server.') 
    parser.add_argument('-c', '--client_name', default='alice', help='Name of the worker client.')
    args = parser.parse_args()
    
    client = Client(
        host = args.host,
        port = int(args.port),
        client_name = args.client_name,
    )

    if args.work:
        asyncio.run(client.worker())
    elif args.reset:
        asyncio.run(client.reset())
        