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
from lib.utilities import valid_log_level

# Run once in the main script
logging.config.dictConfig(cfg.logging_config)

# Include in each module:
LOG = logging.getLogger('CLIENT')


class Client:
    def __init__(self, host: str, port: int, client_name: str) -> None:
        self.url = f'http://{host}:{port}'
        self.client_name = client_name
        self.client_path = f'{cfg.data_path}/clients/{self.client_name}'
        self.sleep_count = cfg.cli_sleep_count
        self.sleep_time = cfg.cli_sleep_time
        self.__post_init__()

    def __post_init__(self) -> None:
        # Inintialize the GeneticExchangeClient tasked with running the tapes
        self.gce = GeneticExchangeClient()

        if not os.path.exists(self.client_path):
            os.makedirs(self.client_path)
        
    async def worker(self, reset: bool = False, rule: str = 'rule_0') -> str:
        '''Get a tape to work on.'''
        try:
            async with aiohttp.ClientSession() as session:
                
                if reset:
                    response = await session.get(f'{self.url}/reset', data='RESET SERVER')
                else:
                    response = await session.get(f'{self.url}/get', data='NEW WORKER')
                
                while response.status == 202:
                    data = await response.read()

                    header = data[:4]
                    tape = data[4:]
                    
                    tape = self.gce.run(tape, rule=rule)
                    response = await session.post(f'{self.url}/post', data=header + tape)
                    LOG.debug(f"Message - {tape[:10]}, response - {response.status} {self.client_name}")

                await self.status_handler(response)
        except aiohttp.client_exceptions.ServerDisconnectedError as e:
            LOG.warning(f'Connection to server lost: {e}, reset={reset}')
            await self._work_waiter()
        except Exception as e:
            LOG.exception(f'worker error={e}, reset={reset}', exc_info=True)
            raise(e)

    async def status_handler(self, response: aiohttp.ClientResponse) -> None:
        '''Handle the response from the server.'''
        msg = await response.text()

        if response.status == 204:
            LOG.info(f'Server temporarily unavailable.')
            await self._work_waiter()
        elif response.status == 400:
            LOG.error(f'ValueError. Status: {response.status} {msg}')
        elif response.status == 408:
            if len(self.gce.times) > 0:
                LOG.info(f'''{self.client_name.upper()} has finished work. 
                Status: {response.status} {msg}
                Runs: {len(self.gce.times)}, Instruction Counts: {np.sum(self.gce.instruction_counts)}
                Total Time: {round(np.sum(self.gce.times), 4)}, Average Time: {round(np.mean(self.gce.times, ), 4)}
                Loop Errors: {self.gce.loop_errors}, Max Read Errors: {self.gce.exaustion_errors}
                ''')
            else:
                LOG.info(f'Server has stopped Status: {response.status} {msg}. To restart request: python3 -m --reset... ')
        else:
            LOG.error(f'Worker Error Status: {response.status} {msg}')

    async def _work_waiter(self) -> None:
        '''Wait for the server to become available.'''
        await asyncio.sleep(self.sleep_time)
        self.sleep_count -= 1
        self.sleep_time += 0.1
        if self.sleep_count == 0:
            LOG.warning('Server is not responding after 10 tries.')
        else:
            await self.worker()


# Example usage
# python3 -m client --work -hs 'localhost' -p 8080 -c alice -v --log-level info -r rule_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a message in bytes format to a server.')
    
    parser.add_argument('--work', action='store_true', help='Start worker.')
    parser.add_argument('--reset', action='store_true', help='Reset Server.')
    parser.add_argument('-hs', '--host', type=str, default=cfg.api_host, help='ip address or "localhost" of server.')  
    parser.add_argument('-p', '--port', type=int, default=cfg.api_port, help='Port of server.') 
    parser.add_argument('-c', '--client_name', type=str, default='alice', help='Name of the worker client.')
    parser.add_argument('-r', '--rule', type=str, default='rule_0', help='Rule to run the tape on.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity.')
    parser.add_argument('--log-level', type=valid_log_level, default='DEBUG', 
                        help=f'Set the logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is {cfg.log_level}.')
    args = parser.parse_args()

    # Logging
    logging.config.dictConfig(cfg.logging_config)
    if args.verbose:
        # Logs to the CLI as well as the log file
        LOG = logging.getLogger('VERBOSE_CLIENT')
    else:
        # Only logs to the log file
        LOG = logging.getLogger('CLIENT')

    LOG.setLevel(args.log_level)
    LOG.debug(f"Client logging is configured: {LOG}.")
    
    # Initialize the Client
    client = Client(
        host = args.host,
        port = args.port,
        client_name = args.client_name,
    )

    if args.work:
        asyncio.run(client.worker(rule=args.rule))
    elif args.reset:
        asyncio.run(client.worker(reset=True, rule=args.rule))
        