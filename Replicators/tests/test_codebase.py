'''Replicators Copyright (C) 2024  RC Bravo Consuling Inc., https://github.com/rcbravo-dev'''

import numpy as np
import pytest
from ..src.lib.codebase import (
    create_gene_pool_pdf,
    GeneticPool,
    GeneticExchangeClient,
    replicator_gene_from_pdf,
    array_to_string,
    determine_gene_fitness,

)


# Create test fixtures
@pytest.fixture
def unfit_pool() -> GeneticPool:
    gp = GeneticPool(512, 64)
    gp.create()
    return gp

@pytest.fixture
def fit_pool() -> GeneticPool:
    gp = GeneticPool(512, 64)
    gp.load('/Users/Shared/SharedProjects/Projects/BiologyCode/Replicators/tests/_testpool_with_high_fitness_genes.npz')
    return gp

@pytest.fixture
def pdf() -> np.ndarray[np.float32]:
    gp = GeneticPool(512, 64)
    gp.load('/Users/Shared/SharedProjects/Projects/BiologyCode/Replicators/tests/_testpool_with_high_fitness_genes.npz')
    return gp.create_gene_pool_pdf()


# Tests
def test_create_gene_pool_pdf(fit_pool: GeneticPool):
    pool_size = fit_pool.pool_size
    tape_length = fit_pool.tape_length

    fit_pool = np.frombuffer(fit_pool.pool, dtype=np.uint8).reshape(pool_size, tape_length).copy()

    # Make the first column have 80% 1s and 20% 2s
    fit_pool[:, 0] = 1
    fit_pool[:int(pool_size * 0.2), 0] = 2 

    pdf = create_gene_pool_pdf(fit_pool, pool_size, tape_length)

    assert np.isclose((pdf[1, 0], pdf[2, 0]), (0.8, 0.2), atol=0.1).all()
    assert pdf.shape == (256, tape_length)
    assert pdf.dtype == np.float16
    assert np.all(pdf >= 0)
    assert np.all(pdf <= 1)
    assert np.allclose(np.sum(pdf, axis=0), np.ones(tape_length))

def test_replicator_gene_from_pdf(pdf):
    gene = replicator_gene_from_pdf(pdf)
    gene_str = array_to_string(gene)

    assert gene.shape == (64,)
    assert gene.dtype == np.uint8
    assert gene_str.isascii()
    assert gene_str == 'w..................................w...........[......<....{..w]'

def test_determine_gene_fitness(pdf):
    gene = replicator_gene_from_pdf(pdf)    

    gene_str, dist = determine_gene_fitness(gene)

    assert gene_str == 'w..................................w...........[......<....{..w]'
    assert dist == 1

def test_genetic_pool(fit_pool: GeneticPool):
    assert fit_pool.size == fit_pool.pool_size * fit_pool.tape_length
    assert len(fit_pool) == fit_pool.pool_size
    assert type(fit_pool.pool) == bytearray

    # Get item should return a tape as a bytearray
    tape0 = fit_pool[0]
    assert type(tape0) == bytearray
    assert len(tape0) == fit_pool.tape_length

    tape1 = np.frombuffer(fit_pool[0], dtype=np.uint8).copy()
    tape1[:5] = 100

    # Assigning a numpy array to a pool should be done with tobytes()
    with pytest.raises(ValueError) as e:
        # Raise type is not bytearray
        fit_pool[0] = tape1
        # Raise length is not tape_length
        fit_pool[0] = tape0[5:]
    # Should not raise
    fit_pool[0] = tape1.tobytes()
    
    # Setting a tape should change the pool
    assert fit_pool[0][:5] == b'ddddd'
    assert fit_pool[0][5:] == tape0[5:]

def test_mutations(unfit_pool):
    pool_0 = np.frombuffer(unfit_pool.pool, dtype=np.uint8).copy()

    unfit_pool.mutation(rate=0.05)

    pool_1 = np.frombuffer(unfit_pool.pool, dtype=np.uint8).copy()

    same_percent = np.sum(pool_0 == pool_1) / len(pool_0)

    assert np.allclose(same_percent, 0.95, atol=0.02, rtol=1e-3), f'Mutation percent={1 - same_percent}'


