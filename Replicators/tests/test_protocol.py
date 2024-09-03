import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# path = [
#     '/Users/Shared/SharedProjects/Projects/BiologyCode/Replicators/src/lib',
#     script_dir,
#     # '/Library/Frameworks/Python.framework/Versions/3.10/lib/python310.zip',
#     '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10',
#     # '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload',
#     '',
#     # '/Users/Shared/SharedProjects/Projects/VirtualEnvironments/sci/lib/python3.10/site-packages'
# ]
# for p in path[::-1]:
#     sys.path.insert(0, p)


# # Add the sibling directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '../src/lib')))

# print()
# print(script_dir)
# print(sys.path, end='\n\n')

import pytest
import numpy as np

# Now Python knows where to find my_module
from codebase import (
    encoder64, 
    decoder64
)
# from src.lib.zka_configurator import configurations as cfg



def test_encoder64():
    # Test the encoder
    a = np.random.randint(0, 255, 64).astype(np.uint8)
    b = encoder64(a)
    c = decoder64(b)
    return np.testing.assert_array_equal(a, c)

print('Test encoder64', test_encoder64())

# # Add a test fixture to create the operator and application
# @pytest.fixture
# def ops():
#     ops = ZeroKnowledgeProtocolOperator()
#     return ops

# @pytest.fixture
# def api():
#     api = ZeroKnowledgeProtocolApplication()
#     return api

# @pytest.fixture
# def osd(ops):
#     osd = ops.genesis()
#     return osd

# @pytest.fixture
# def dbr(api, ops):
#     dbr = ops.dbr
#     api.genesis(dbr)
#     dbr['one_time_pad'] = api.otp
#     return dbr

# # PIN FIXTURES
# @pytest.fixture
# def ops_osd_with_pin(ops, pin='1234'):
#     osd = ops.genesis(pin=pin)
#     return ops, osd

# @pytest.fixture
# def dbr_with_pin(api, ops_osd_with_pin):
#     ops, _ = ops_osd_with_pin
#     dbr = ops.dbr
#     api.genesis(dbr)
#     dbr['one_time_pad'] = api.otp
#     return dbr


# # NEW TESTS 
# '''Need to update the tests to reflect the new changes to the protocol: sec_size in init'''
# def test_genesis(ops, osd):
#     # Test the dbr message
#     assert isinstance(ops.message, bytes)

#     # Test the verify message
#     assert isinstance(ops.verify_array, bytes)

#     # Test the osd
#     for v in osd.values():
#         assert v.dtype == np.uint8

# def test_challenge(ops, api, osd, dbr):
#     c = ops.create_challenge()

#     ops.challenge(osd, c)
#     api.challenge(dbr, c)

#     np.testing.assert_array_equal(ops.binary_secret_hash, api.binary_secret_hash)
#     np.testing.assert_array_equal(ops.cookie_a, api.cookie_a)
#     np.testing.assert_array_equal(ops.cookie_b, api.cookie_b)
#     np.testing.assert_array_equal(ops.verify_array, api.verify_array)

# def test_msv(ops, api, osd, dbr):
#     # Ops creates the verification_hash
#     ops.master_secret_verification(osd)
    
#     # API creates the name and name_cookie
#     api.master_secret_verification(dbr, osd['secret'])

#     # Create the verification hash for the API
#     np.testing.assert_array_equal(ops.verify_array, api.verify_array)

# def test_genesis_with_pin(ops, ops_osd_with_pin):
#     ops, osd = ops_osd_with_pin

#     # Test the dbr message
#     assert isinstance(ops.message, bytes)

#     # Test the verify message
#     assert isinstance(ops.verify_array, bytes)

#     # Test the osd
#     for v in osd.values():
#         assert v.dtype == np.uint8

# def test_challenge_with_pin(ops, api, ops_osd_with_pin, dbr_with_pin):
#     ops, osd = ops_osd_with_pin

#     c = ops.create_challenge()

#     ops.challenge(osd, c)
#     api.challenge(dbr_with_pin, c)

#     np.testing.assert_array_equal(ops.binary_secret_hash, api.binary_secret_hash)
#     np.testing.assert_array_equal(ops.cookie_a, api.cookie_a)
#     np.testing.assert_array_equal(ops.cookie_b, api.cookie_b)

# def test_msv_with_pin(ops, api, ops_osd_with_pin, dbr_with_pin, pin='1234'):
#     ops, osd = ops_osd_with_pin

#     # Ops creates the verification_hash
#     ops.master_secret_verification(osd)
    
#     # API creates the name and name_cookie
#     api.master_secret_verification(dbr_with_pin, osd['secret'], pin=pin)

#     # Create the verification hash for the API
#     np.testing.assert_array_equal(ops.verify_array, api.verify_array)

# def test_msv_with_pin_fail(ops, api, ops_osd_with_pin, dbr_with_pin, pin='5678'):
#     ops, osd = ops_osd_with_pin

#     # Ops creates the verification_hash
#     ops.master_secret_verification(osd)
    
#     # API creates the name and name_cookie
#     api.master_secret_verification(dbr_with_pin, osd['secret'], pin=pin)

#     # Create the verification hash for the API (should not be equal)
#     assert not np.array_equal(ops.verify_array, api.verify_array)

# def test_uniform_distribution(ops):
#     # Test the arrays are in the uniform distribution
#     c = 1000
#     r = []
#     for _ in range(c):
#         a = ops.random_array(64)
#         b = ops.hash_array(a)
#         c = ops.shake_array(a, 64)
#         r.append(np.sum(a) / 64)
#         r.append(np.sum(b) / 64)
#         r.append(np.sum(c) / 64)
#     s = np.sum(r) / (len(r))

#     np.testing.assert_approx_equal(s, 127.5, significant=2)

# def test_abc_methods(ops):
#     # Test Challenge rotates secret properly
#     s = ops.random_array(64)
#     c = ops.create_challenge()
#     t = ops.rotate_secret(s, c, binary=False)
#     t = ops.rotate_secret(t, c, binary=False)
#     # Rotate twice should rotate back to original
#     np.testing.assert_array_equal(s, t)
    
#     # Test Challenge rotates secret properly (binary)
#     s = ops.array_to_binary(s)
#     t = ops.rotate_secret(s, c, binary=True)
#     t = ops.rotate_secret(t, c, binary=True)
#     # Rotate twice should rotate back to original (binary)
#     np.testing.assert_array_equal(s, t)  

#     # Test the split
#     a = ops.random_array(64)
#     b, c = ops.cookies_ab(a)
#     assert len(b) == 64
#     assert len(c) == 64  

#     # Test that arrays are not equal
#     assert not np.array_equal(a, b)
#     assert not np.array_equal(a, c)
#     assert not np.array_equal(b, c)

#     # # Test verification hash
#     # n = ops.random_array(32)
#     # nc = ops.random_array(64)
#     # a = ops.create_verification_hash(n, nc, valid=True)
#     # b = ops.create_verification_hash(n, nc, valid=False)
#     # assert not np.array_equal(a, b)
