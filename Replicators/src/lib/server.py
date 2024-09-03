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
from lib.codebase import PoolServer, encoder64, decoder64
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


class ServerABC:
    def __init__(self, host: str = 'localhost', port: int = 9999) -> None:
        self.host = host
        self.port = port
        self.cnt = 10
        self.rng = np.random.default_rng()

    def run_server(self):
        app = web.Application()
        app.router.add_post('/', self.handler)
        app.router.add_post('/stream', self.stream_handler)

        web.run_app(app, host=self.host, port=self.port)

    async def handler(self, request):
        # Get the raw byte data from the request
        byte_data = await request.read()  

        try:
            print('Route init: ', byte_data)

            arr = np.arange(10, dtype=np.uint8)
            
            return web.Response(body=arr.tobytes(), status=202)
        
        except Exception as e:
            LOG.error(f"Error in handler: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')

    async def stream_handler(self, request):
        # Get the raw byte data from the request
        byte_data = await request.read()  

        try:
            print('Route Stream: ', byte_data)

            arr = np.frombuffer(byte_data, dtype=np.uint8).copy()
            arr -= 1
            
            # Introduce random errors
            if self.rng.random() < 0.2:
                raise ValueError('Random error')

            if self.cnt > 0:
                self.cnt -= 1
                return web.Response(body=arr.tobytes(), status=202)
            else:
                self.cnt = 10
                return web.Response(status=200, text='Done')     
        
        except ValueError as e:
            LOG.error(f"Error in stream_handler: {e}", exc_info=True)
            # Text cannot be 'e' because it will make the response a 500 error
            return web.Response(status=403, text='Random error')
        

class ServerMain:
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 9999,
                 epochs: int = 10, 
                 pool_size: int = 100, 
                 tape_length: int = 64, 
                 experiment_name: str = 'run_001',
                 first_run: bool = True,
                 track_generations: bool = False,
                 ) -> None:      
        self.host = host
        self.port = port
        self.epochs = epochs
        self.pool_size = pool_size
        self.tape_length = tape_length
        self.experiment_name = experiment_name
        self.first_run = first_run
        self.track_generations = track_generations
        self.data_path = cfg.data_path
        self.experiment_path = f'{self.data_path}/server/{experiment_name}/'
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

    def initialize_server(self):
        self.ps = PoolServer(self.epochs, self.pool_size, self.tape_length, self.experiment_path)

        if self.first_run:
            self.ps.pool.create()
            self.ps.pool.save(overwrite=True)
        else:
            self.ps.pool.load(most_recent=True)
    
    def run_server(self):
        self.initialize_server()

        app = web.Application()
        app.router.add_post('/', self.handler)
        app.router.add_post('/stream', self.stream_handler)

        web.run_app(app, host=self.host, port=self.port)

    async def handler(self, request):
        # Get the raw byte data from the request
        try:
            header = await request.read()  
            
            if header == b'NEW WORKER':
                msg = await self.ps.serve()
            else:
                # print(f'Invalid header: {header}')
                raise ValueError(f'Invalid header: {header}')

            if msg == 'STOP':
                raise StopIteration()
            
        except StopIteration:
            LOG.info('All tapes have been served')
            return web.Response(status=204, text='Server has stopped')
        except ValueError as e:
            LOG.error(f"Error in handler: {e}", exc_info=True)
            return web.Response(status=400, text=f'Invalid header: {header}')
        except Exception as e:
            LOG.error(f"Error in handler: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return web.Response(status=202, body=msg)

    async def stream_handler(self, request):
        # Get the raw byte data from the request
        try:
            modified_tape = await request.read()  
            
            await self.ps.receive(modified_tape)

            msg = await self.ps.serve()

            if msg == 'STOP':
                raise StopIteration()
            
        except StopIteration:
            LOG.info(f'All tapes have been served. Epochs: {self.ps.epochs}')
            
            if self.track_generations:
                self.ps.pool.save(overwrite=False)
            else:
                self.ps.pool.save(overwrite=True)

            LOG.info(f'Pool saved. Compression ratio: {self.ps.pool.compression_ratio}')
            
            return web.Response(status=204, text='Server has stopped')
        except TimeoutError as e:
            LOG.warning(f"Error in stream_handler: {e}")
            return web.Response(status=408, text='Timeout warning')
        except Exception as e:
            LOG.error(f"Error in stream_handler: {e}", exc_info=True)
            return web.Response(status=500, text='Server error')
        else:
            return web.Response(status=202, body=msg)
        
    # This is on the server
    async def route_request_v1(self, request):
        data = await request.text()
        header = decoder64(data)[0]
        
        if header == 10:
            print('Transmitting tapes', header)
            return await self.transmit_tapes()
        elif header == 20:
            return await self.receive_tapes(request)
        elif header == 30:
            print('receive_transmit tapes', header)
            return await self.receive_transmit(request)
        else:
            return web.Response(status=400, text=f"Invalid header: {header}")
            
    async def transmit_tapes_v1(self):
        '''Return the next tape in the pool'''
        try:
            msg = await self.ps.serve()
        except Exception as e:
            LOG.error(f"Error in transmit_tapes: {e}")
            return web.Response(status=500, text=e)
        else:
            return web.Response(text=msg)
    
    async def receive_tapes_v1(self, request):
        '''Recieve a tape from the client'''
        try:
            tape = await request.text()

            await self.ps.receive(tape[1:])
            msg = 'ACK'
        except Exception as e:
            LOG.error(f"Error in receive_tapes: {e}")
            return web.Response(status=500, text=e)
        else:
            return web.Response(text=msg)

    async def receive_transmit_v1(self, request):
        '''Recieve a tape from the client and return the next tape in the pool'''
        try:
            tape = await request.text()

            await self.ps.receive(tape)

            msg = await self.ps.serve()
        except Exception as e:
            LOG.error(f"Error in receive_transmit: {e}", exc_info=True)
            return web.Response(status=500, text=e)
        else:
            return web.Response(text=msg)
        
      

if __name__ == "__main__":
    # python3 -m server --run_server -hs 'localhost' -p 9999 -e 10 -ps 100 -t 64 -en 'run_001' --first_run --track_gen

    parser = argparse.ArgumentParser(description='Server the GeneticCode.')
    parser.add_argument('--run_server', action='store_true', help='Initialize the server and run forever.')
    parser.add_argument('-hs', '--host', default='localhost', help='ip address or "localhost" of server.')  
    parser.add_argument('-p', '--port', default=9999, help='Port of server.') 
    parser.add_argument('-e', '--epoch', default=10, help='Number of times to run all tapes.')
    parser.add_argument('-ps', '--pool_size', default=100, help='Number of tapes in the pool.')
    parser.add_argument('-t', '--tape_length', default=64, help='Number of bytes in the tape.')
    parser.add_argument('-en', '--experiment_name', default='run_001', help='Name of the genetic pool.')
    parser.add_argument('--first_run', action='store_true', help='Initialize the server and run forever.')
    parser.add_argument('--track_gen', action='store_true', help='Initialize the server and run forever.')

    args = parser.parse_args()

    if args.first_run:
        first_run = True
    else:
        first_run = False

    if args.track_gen:
        track_generations = True
    else:
        track_generations = False

    sm = ServerMain(
        host = args.host,
        port = int(args.port),
        epochs = int(args.epoch),
        pool_size = int(args.pool_size),
        tape_length = int(args.tape_length),
        experiment_name = args.experiment_name,
        first_run = first_run,
        track_generations = track_generations,
    )
    
    LOG.info(f'Server [{args.experiment_name}] initialized on {args.host}:{args.port}')
    sm.run_server() 
