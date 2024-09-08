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
    gp.load('./_testpool_with_high_fitness_genes')
    return gp

@pytest.fixture
def pdf() -> np.ndarray[np.float32]:
    gp = GeneticPool(512, 64)
    gp.load('./_testpool_with_high_fitness_genes')
    return create_gene_pool_pdf(gp.pool, 512, 64)


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
    assert pdf.dtype == np.float32
    assert np.all(pdf >= 0)
    assert np.all(pdf <= 1)
    assert np.allclose(np.sum(pdf, axis=0), np.ones(tape_length))

def test_replicator_gene_from_pdf(pdf):
    gene = replicator_gene_from_pdf(pdf, 64)
    gene_str = array_to_string(gene)

    assert gene.shape == (64,)
    assert gene.dtype == np.uint8
    assert gene_str.isascii()
    assert gene_str == 'ww[w....................{.....................<......]..........'

def test_determine_gene_fitness(pdf):
    gene = replicator_gene_from_pdf(pdf, 64)    

    gene_str, dist = determine_gene_fitness(gene)

    assert gene_str == 'ww[w....................{.....................<......]..........'
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