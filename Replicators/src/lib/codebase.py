'''Replicators Copyright (C) 2024  RC Bravo Consuling Inc., https://github.com/rcbravo-dev'''

import secrets
import numpy as np
import matplotlib.pyplot as plt
import base64
import time
from pathlib import Path
import asyncio


random_array = lambda size: np.array([secrets.randbits(8) for _ in range(size)], dtype=np.uint8)

# Mapping instructions to characters
instruction_mapping = {
    122: '>',  # Move head 0 forward
    133: '<',  # Move head 0 backward
    123: '}',  # Move head 1 forward
    132: '{',  # Move head 1 backward
    124: '+',  # Increment value at head 0
    131: '-',  # Decrement value at head 0
    125: '^',  # Increment value at head 1
    130: '_',  # Decrement value at head 1
    126: 'r',  # Copy value from head 1 to head 0 or vice versa
    129: 'w',  # Move value from head 0 to head 1 or vice versa
    127: '[',  # Loop start
    128: ']',  # Loop end
}

# Reverse mapping
instruction_mapping_reverse = {v: k for k, v in instruction_mapping.items()}

rng = np.random.default_rng()

def array_to_string(arr: np.ndarray[np.uint8]) -> str:
    return ''.join([instruction_mapping.get(x, '.') for x in arr])

def string_to_array(s: str, null_byte: int = 1) -> np.ndarray[np.uint8]:
    return np.array([instruction_mapping_reverse.get(x, null_byte) for x in s], dtype=np.uint8)

def create_gene_pool_pdf(pool: np.ndarray[np.uint8], pool_size: int, tape_length: int) -> np.ndarray[np.float16]:
    '''Returns a probability distribution of the gene pool, by position on the tape'''
    if isinstance(pool, (bytes, bytearray)):
        pool = np.frombuffer(pool, dtype=np.uint8)
    
    pool = pool.reshape(pool_size, tape_length).copy()
    matrix = np.zeros([256, tape_length], dtype=np.float16)

    for i in range(256):
        matrix[i, :] = np.count_nonzero(pool == i, axis=0)

    return matrix / pool_size

def replicator_gene_from_pdf(pdf: np.ndarray) -> np.ndarray[np.uint8]:
    '''Returns a gene that can replicate the tape'''
    tape_length = pdf.shape[1]

    gene = np.zeros(tape_length, dtype=np.uint8)
    
    for i in range(tape_length):
        gene[i] = np.argmax(pdf[:, i])

    return gene

def find_dominate_gene(pool: np.ndarray[np.uint8], pool_size: int, tape_length: int) -> np.ndarray[np.uint8]:
    '''Returns the dominate gene in the gene pool'''
    # Generate a PDF for the gene pool
    pdf = create_gene_pool_pdf(pool, pool_size, tape_length)
    
    # Find the dominate gene
    return replicator_gene_from_pdf(pdf)

def determine_gene_fitness(gene: np.ndarray[np.uint8]) -> tuple[str, float]:
    # Determine the genes ability to replicate a random tape
    gce = GeneticExchangeClient()
    
    tape_length = len(gene)

    random_tape = rng.integers(0, 256, tape_length, dtype=np.uint8)
    
    tape = gce.run(np.concatenate([gene, random_tape], dtype=np.uint8).tobytes())

    arr1 = np.frombuffer(tape[:tape_length], dtype=np.uint8)
    arr2 = np.frombuffer(tape[tape_length:], dtype=np.uint8)
    
    # dist = cosine_similarity(arr1, arr2)

    dist = hamming_similarity(arr1, arr2)

    return array_to_string(gene), dist

def hamming_similarity(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> int:
    '''Returns the hamming distance between two arrays'''
    x = a ^ b
    return 1 - np.sum(np.unpackbits(x)) / (len(a) * 8)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    '''Returns the cosine distance between two arrays'''
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def plot_pdf(pdf: np.ndarray[np.uint16]):
    '''Plots the probability distribution of the gene pool'''
    plt.imshow(pdf, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.xlabel('Position')
    plt.ylabel('Byte Value')
    plt.title('Byte Value Probability Distribution by Position')
    plt.show()

def encoder64(a: np.ndarray[np.uint8] | bytes) -> str:
    '''Converts a numpy array of uint8 to a base64 encoded string'''
    return base64.b64encode(a).decode()

def decoder64(b: str | bytes) -> np.ndarray[np.uint8]:
    '''Converts a base64 encoded string to a numpy array of uint8'''
    return np.frombuffer(base64.b64decode(b), dtype=np.uint8)

def calculate_crack_time(t: float, parallel_computers: int, bits: int) -> float:
    sec_per_year = 60 * 60 * 24 * 365
    guesses_per_second = (1 / t) * parallel_computers
    sec_to_try_all = 2**bits / guesses_per_second
    years = sec_to_try_all / sec_per_year
    if years > 1:
        return int(years), f'Estimated time to crack: {years:,.0f} years'
    else:
        return years, f'Estimated time to crack: {years * 365:,.2f} days'

def array_to_int(arr: np.ndarray[np.uint8]) -> int:
    a = np.unpackbits(arr)
    b = ''.join([str(x) for x in a])
    return int(b, 2)

def large_int_to_uint8_array(num: int):
    # Convert the integer to bytes
    num_bytes = num.to_bytes((num.bit_length() + 7) // 8, 'big')
    # Convert the bytes to a numpy array
    return np.frombuffer(num_bytes, dtype=np.uint8)

def hex_to_array(hex_str: str) -> np.ndarray[np.uint8]:
    return np.array([x for x in bytes.fromhex(hex_str)], dtype=np.uint8)

def create_head_position_lookup_table(mod: int):
    '''Returns a lookup table for the mod function and keeps 
    each head within the bounds of their individual tape'''
    pos_ = [0 + x for x in range(mod)]
    add_ = [(x + 1) % mod for x in pos_]
    sub_ = [(x - 1) % mod for x in pos_]
    
    pos = [x + mod for x in pos_]
    add = [x + mod for x in add_]
    sub = [x + mod for x in sub_]

    return add_ + add, pos_ + pos, sub_ + sub

def create_modulo_lookup_table(mod: int):
    '''Returns a lookup table for the mod function'''
    pos_ = [0 + x for x in range(mod)]
    add_ = [(x + 1) % mod for x in pos_]
    sub_ = [(x - 1) % mod for x in pos_]
    
    return add_, pos_, sub_

def head_position(table: tuple, head: int, add: bool = False, sub: bool = False):
    '''Returns the new head position'''
    add_, pos_, sub_ = table
    if add:
        return add_[head]
    elif sub:
        return sub_[head]
    else:
        return pos_[head]

def visualize_tape(i, h0, h1, tape, sleep_time: float = 0.0, mapping: dict = instruction_mapping):
    # ANSI color codes for highlighting
    GREEN_BG = '\033[42m'
    RED_BG = '\033[41m'
    YELLOW_BG = '\033[43m'
    RESET = '\033[0m'

    output = []
    for index, byte in enumerate(tape):
        char = mapping.get(byte, '.')
        
        if index == i:
            char = f'{GREEN_BG}{char}{RESET}'
        elif index == h0:
            char = f'{RED_BG}{char}{RESET}'
        elif index == h1:
            char = f'{YELLOW_BG}{char}{RESET}'

        output.append(char)
    
    # * 5 prevents extra characters from being printed at the end
    print('\r' + ''.join(output) + ' ' * 5, end='', flush=True)

    # Add a small delay for better visualization
    time.sleep(sleep_time)  
   
    
class GeneticExchangeClient:
    def __init__(self):
        self.instruction_counts = []
        self.exaustion_errors = 0
        self.loop_errors = 0
        self.times = []
        self.instruction_set = {k for k in instruction_mapping.values()}
        self.first_run = True
        self.max_reads = 6000
    
    def __setattr__(self, name, value):
        if name == 'tape_length':
            if value % 2 == 0:
                self.tape_midpoint = value // 2
            else:
                raise ValueError('Tape length must be an even number')
        
        super().__setattr__(name, value)

    def execute_instruction(self, inst: str, i: int, h0: int, h1:int, tape: np.ndarray) -> tuple:
        '''Executes the instruction and returns the updated state'''
        if inst in {'.', '<', '>', '{', '}', '+', '-', '^', '_', 'r', 'w'}:
            if inst == '.':
                pass

            # Moving header 0
            elif inst == '>':
                h0 = head_position(self.head_table, h0, add=True)
            elif inst == '<':
                h0 = head_position(self.head_table, h0, sub=True)
            
            # Moving header 1
            elif inst == '}':
                h1 = head_position(self.head_table, h1, add=True)
            elif inst == '{':
                h1 = head_position(self.head_table, h1, sub=True)

            # Transforming values
            elif inst == '+':
                tape[h0] = head_position(self.mod_table, tape[h0], add=True)
            elif inst == '-':
                tape[h0] = head_position(self.mod_table, tape[h0], sub=True)
            elif inst == '^':
                tape[h1] = head_position(self.mod_table, tape[h1], add=True)
            elif inst == '_':
                tape[h1] = head_position(self.mod_table, tape[h1], sub=True)

            # Read from
            elif inst == 'r':
                if i < self.tape_midpoint:
                    tape[h0] = tape[h1]
                else:
                    tape[h1] = tape[h0]
            
            # Write to
            elif inst == 'w':
                if i < self.tape_midpoint:
                    tape[h1] = tape[h0] 
                else:
                    tape[h0] = tape[h1]

            # Move instruction pointer to the next instruction. Loops are handled separately
            i += 1
        
        # Loops
        elif inst == '[':
            # matching_bracket = np.where(tape[i:] == 111)
            if tape[i + 1] == 0:
                # Loop is over, move to the matching bracket or raises if not found
                i = i + np.where(tape[i:] == 128)[0][0]
            else:
                # Loop is not over or unbounded, move to the next instruction
                i += 1

        elif inst == ']':
            # This should raise if no matching bracket is found
            chk = np.where(tape[:i] == 127)[0][-1]
            
            if tape[chk + 1] == 0:
                # If the loop is at 0, then the inst pointer does not move to the beginning of the loop,
                # but rather to the next instruction
                i += 1
            else:
                # Else move the pointer to the matching bracket, to start loop over
                i = chk

        return i, h0, h1, tape

    def run(self, tape: bytes) -> bytes:
        tape = np.frombuffer(tape, dtype=np.uint8).copy()

        if self.first_run:
            self.tape_length = len(tape)
            self.head_table = create_head_position_lookup_table(self.tape_midpoint)
            self.mod_table = create_modulo_lookup_table(256)
            self.first_run = False
        else:
            if len(tape) != self.tape_length:
                raise ValueError(f'Tape length has changed. Expected {self.tape_length}, received {len(tape)}')

        try:
            t0 = time.time()
            h0, h1 = 0, self.tape_midpoint
            i, cnt, reads = 0, 0, 0

            while True:
                inst = instruction_mapping.get(tape[i], '.')

                i, h0, h1, tape = self.execute_instruction(inst, i, h0, h1, tape)
                
                if inst in self.instruction_set:
                    cnt += 1

                if reads < self.max_reads:
                    reads += 1
                else:
                    raise StopIteration(f'Read count exceeded={reads}')

        except IndexError:
            if i == self.tape_length:
                pass
            else:
                self.loop_errors += 1
        except StopIteration:
            self.exaustion_errors += 1
        except Exception as e:
            raise e
        finally:
            self.instruction_counts.append(cnt)
            self.times.append(time.time() - t0)

            return tape.tobytes()
        
    def visualize(self, tape: bytes, sleep_time: float = 0.2) -> bytes:
        tape = np.frombuffer(tape, dtype=np.uint8).copy()

        if self.first_run:
            self.tape_length = len(tape)
            self.head_table = create_head_position_lookup_table(self.tape_midpoint)
            self.mod_table = create_modulo_lookup_table(256)
            self.first_run = False
        else:
            if len(tape) != self.tape_length:
                raise ValueError(f'Tape length has changed. Expected {self.tape_length}, received {len(tape)}')

        instruction_list = []
        try:
            t0 = time.time()
            h0, h1 = 0, self.tape_midpoint
            i, cnt, reads = 0, 0, 0

            # Print the initial state
            visualize_tape(i, h0, h1, tape)

            while True:
                inst = instruction_mapping.get(tape[i], '.')

                i, h0, h1, tape = self.execute_instruction(inst, i, h0, h1, tape)
                
                if inst in self.instruction_set:
                    cnt += 1
                    instruction_list.append(inst)
                
                if reads < self.max_reads:
                    reads += 1
                else:
                    raise StopIteration(f'Read count exceeded={reads}')
    
                # Visualize the tape after each step
                visualize_tape(i, h0, h1, tape, sleep_time=sleep_time)

        except IndexError:
            if i == self.tape_length:
                print('\nEnd of tape reached')
            else:
                self.loop_errors += 1
                print('\nloop_errors', i, h0, h1, inst)
        except StopIteration as e:
            print(f'\n{e}')
            self.exaustion_errors += 1
        except Exception as e:
            print(f'Run Error: {e}')
        finally:
            self.instruction_counts.append(cnt)
            self.times.append(time.time() - t0)
            # List of all instructions executed
            self.instruction_list = ''.join(instruction_list)

            return tape


class GeneticPoolABC:
    def __init__(self, pool_size: int = 100, tape_length: int = 64, filename: str = ''):
        self.pool_size = pool_size
        self.tape_length = tape_length
        self.size = pool_size * tape_length
        if filename:
            self.filename = f'{filename}genetic_pool'
        else:
            self.filename = 'genetic_pool'
        self.rng = np.random.default_rng()

    def __setattr__(self, name, value):
        if name == 'pool_size':
            if value > 65535:
                raise ValueError('Pool size must be less than or equal to 65535')
        elif name == 'tape_length':
            if value % 2 != 0:
                raise ValueError('Tape length must be an even number')
        elif name == 'pool_size':
            if value % 2 != 0:
                raise ValueError('Pool size must be an even number')
        
        super().__setattr__(name, value)

    def __getitem__(self, key: int) -> bytes:
        start = key * self.tape_length
        stop = start + self.tape_length
        return self.pool[start:stop]

    def __setitem__(self, key: int, value: bytes):
        if len(value) != self.tape_length:
            raise ValueError('Value length must match tape length')
        elif not isinstance(value, bytes):
            raise ValueError(f'Value must be of type bytes. Received {type(value)}')
        else:
            start = key * self.tape_length
            stop = start + self.tape_length

            self.pool[start:stop] = value

    def __len__(self):
        return self.pool_size
    
    def __sizeof__(self):
        return self.size

    def __iter__(self):
        for i in range(self.pool_size):
            start = i * self.tape_length
            stop = start + self.tape_length
            yield self.pool[start:stop]

    def __call__(self):
        self._tape_to_ascii()

        ps = self.pool_string
        tl = self.tape_length
        
        # Print 64 characters per line
        for i in range(0, len(ps), tl):
            print(ps[i:i+tl])

    def _tape_to_ascii(self):
        t = []
        inst_cnt = 0
        for p in self.pool:
            inst = instruction_mapping.get(p, '.')
            
            if inst == '.':
                pass
            else:
                inst_cnt += 1
            
            t.append(inst)
        
        self.instruction_ratio = inst_cnt / self.size

        self.pool_string = ''.join(t)
    
    def _find_new_path(self, new_path: Path):
        suffix = new_path.suffix
        counter = 1
        while new_path.exists():
            new_path = Path(f"{self.filename}_{counter}{suffix}")
            counter += 1
        return new_path
    
    def create(self):
        pool = np.array([secrets.randbits(8) for _ in range(self.size)], dtype=np.uint8).tobytes()
        self.pool = bytearray(pool)
    
    def save(self, overwrite: bool = False, safe: bool = True):
        new_path = Path(f'{self.filename}_0.npz')
        
        # Make the directory if it does not exist
        if new_path.parent.exists() is False:
            new_path.parent.mkdir(parents=True)

        # If False, then find a new path
        elif new_path.exists() and overwrite is False:
            new_path = self._find_new_path(new_path)

        # If True, then overwrite the existing file
        elif new_path.exists() and overwrite is True:
            if safe:
                x = input(f'Are you sure you want to overwrite the existing file: {new_path}? (y/n): ').lower()
                if x == 'n':
                    new_path = self._find_new_path(new_path)
            else:
                pass

        np.savez_compressed(new_path, self.pool)

        self.compression_ratio = new_path.stat().st_size / self.size

    def load(self, file_path: str) -> None:
        '''Loads the filename if provided intothe genetic pool'''
        path = Path(file_path)

        if path.exists():
            pool = np.load(file_path)['arr_0']

            if len(pool) != self.size:
                raise ValueError('Loaded pool size does not match the expected size')
            else:
                print(f'Loading: {file_path}')
                self.pool = bytearray(pool.tobytes())
        else:
            raise FileNotFoundError(f'No files found: {path}')


class GeneticPool(GeneticPoolABC):
    def __init__(self, pool_size: int = 100, tape_length: int = 64, filename: str = ''):
        super().__init__(pool_size, tape_length, filename)

    def mutation(self, rate: float = 0.05) -> float:
        '''Mutates the genetic pool by a given rate. Rate is 
        the probability of mutation'''
        pool = np.frombuffer(self.pool, dtype=np.uint8)
        add = 1.0 - (rate / 2)
        sub = 0.0 + (rate / 2)

        # Create the mask
        mask = self.rng.random(pool.size)
        mask_add = mask > add
        mask_sub = mask < sub

        # Mutate the pool
        pool[mask_add] += 1
        pool[mask_sub] -= 1

        self.pool = bytearray(pool.tobytes())

        # Return the actual mutation rate
        return (np.sum(mask_add) + np.sum(mask_sub)) / pool.size

    def find_dominate_gene(self) -> np.ndarray[np.uint8]:
        '''Returns the dominate gene in the gene pool'''
        pool = np.frombuffer(self.pool, dtype=np.uint8)

        return find_dominate_gene(pool, self.pool_size, self.tape_length)

    def determine_gene_fitness(self, gene: np.ndarray[np.uint8]) -> tuple[str, float]:
        '''Determines the fitness of a gene to replicate a random tape'''
        return determine_gene_fitness(gene)

    def create_gene_pool_pdf(self) -> np.ndarray[np.float16]:
        '''Returns a probability distribution of the gene pool, by position on the tape'''
        pool = np.frombuffer(self.pool, dtype=np.uint8)

        return create_gene_pool_pdf(pool, self.pool_size, self.tape_length)

    def replicator_gene_from_pdf(self, pdf: np.ndarray[np.uint16]) -> np.ndarray[np.uint8]:
        '''Returns a gene that can replicate the tape'''
        return replicator_gene_from_pdf(pdf)

    def load_most_recent(self):
        path = Path(f'{self.filename}_1.npz')
        suffix = path.suffix

        counter = 1
        while True:
            if path.exists():
                counter += 1
                path = Path(f"{self.filename}_{counter}{suffix}")
            else:
                path = Path(f"{self.filename}_{counter-1}{suffix}")
                break

        if path.exists():
            self.load(path)
        else:
            raise FileNotFoundError(f'No files found: {path}')
        
    def plot_pdf(self):
        '''Plots the probability distribution of the gene pool'''
        plot_pdf(self.create_gene_pool_pdf())


class PoolServer:
    def __init__(self, epochs: int = 3, pool_size: int = 10, tape_length: int = 8, experiment_path: str = 'genetic_pool'):
        self.epochs = int(epochs)
        self.pool_size = int(pool_size)
        self.tape_length = int(tape_length)
        self.experiment_path = experiment_path
        self.pool = GeneticPool(self.pool_size, self.tape_length, experiment_path)
        self.rng = np.random.default_rng()
        self.epochs_plan()
        self.epoch = None
        self.ready = False
        self.sleep_time = 0.05

    def update_pool(self):
        tx = b''
        for t in self.containerTX:
            tx += t

        self.pool.pool = np.frombuffer(tx, dtype=np.uint8)
        
    def __setattr__(self, name, value):
        if name == 'pool_size':
            if value > 65535:
                raise ValueError('Pool size must be less than or equal to 65535')
        elif name == 'tape_length':
            if value % 2 != 0:
                raise ValueError('Tape length must be an even number')
        elif name == 'pool_size':
            if value % 2 != 0:
                raise ValueError('Pool size must be an even number')
        
        super().__setattr__(name, value)

    def init_containers(self):
        self.containerTX = []
        self.containerRX = []
        
        for i in range(self.pool_size):
            arr = self.pool[i].tobytes()
            self.containerTX.append(arr)

        self.ready = True

    async def swap_containers(self):
        if len(self.containerRX) == self.pool_size:
            self.containerTX = self.containerRX.copy()
            self.containerRX = []
        else:
            # Raises an error if the containerRX is not full after 10 tries
            await self.full_container_waiter()

    async def full_container_waiter(self):
        '''Total wait time:
            sleep_time = 0.1 -> 4.5 seconds
            sleep_time = 0.05 -> 2.25 seconds
        '''
        count = 0

        while len(self.containerRX) < self.pool_size:
            if count > 10:
                raise TimeoutError('ContainerRX not full after 10 attempts')
            
            count += 1
            await asyncio.sleep(self.sleep_time * count)

        await self.swap_containers()

    def epochs_plan(self):
        self.server_plan = {}
        
        for i in range(self.epochs):
            # iinfo(min=0, max=65535, dtype=uint16)
            self.server_plan[i] = self.rng.permutation(self.pool_size).astype(np.uint16)

    async def serve_epoch(self, epoch: int):
        for i, j in zip(self.server_plan[epoch][0::2], self.server_plan[epoch][1::2]):
            tape = self.containerTX[i] + self.containerTX[j]
            yield tape
            await asyncio.sleep(0)

    async def serve(self) -> bytes:
        # First Request
        try:
            if self.epoch is None:
                self.epoch = 0
                self.init_containers()
                self.svr = self.serve_epoch(self.epoch)

            return await self.svr.__anext__()
        except StopAsyncIteration:
            self.epoch += 1

            if self.epoch >= self.epochs:
                await self.swap_containers()
                self.ready = False
                self.update_pool()
                return 'STOP'
            else:
                await self.swap_containers()
                self.svr = self.serve_epoch(self.epoch)
                return await self.serve()

    async def receive(self, tape: bytes):
        # t = decoder64(tape)
        # print('Received tape:', len(tape), tape[:10])

        self.containerRX.append(tape[:self.tape_length])
        self.containerRX.append(tape[self.tape_length:])

  
class ValidationError(Exception):
    '''Raised when the verification of a local and 
    remote array, string, or hash fails.'''
    pass

class CounterPartyError(Exception):
    '''Raised when the verification of the user being 
    challenged fails or times out.'''
    pass