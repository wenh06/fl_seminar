# coding=utf-8
# Copyright 2020 Maruan Al-Shedivat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for quadratic objective functions."""

import jax.numpy as jnp
import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random, vmap

from .quadratic import (
    LeastSquares,
    Quadratic,
    create_global_least_squares,
    create_global_quadratic,
    create_random_least_squares,
    create_random_quadratics,
)

FLAGS = flags.FLAGS


class QuadraticTests(absltest.TestCase):
    @staticmethod
    def exact_grad(obj: Quadratic, x: jnp.ndarray) -> jnp.ndarray:
        """Returns exact gradient of the given objective at `x`."""
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(vmap(lambda x: jnp.dot(obj.A, x) - obj.b)(x))

    def test_quadratic_eval(self):
        obj = Quadratic(
            A=jnp.asarray([[2.0, 1.0], [1.0, 2.0]]), b=jnp.asarray([1.0, 2.0])
        )
        x1 = jnp.asarray([1.0, 0.0])
        x2 = jnp.asarray([0.0, 1.0])

        np.testing.assert_allclose(obj(x1), 0.0)
        np.testing.assert_allclose(obj(x2), -1.0)
        np.testing.assert_allclose(obj(jnp.stack([x1, x2])), [0.0, -1.0])

    def test_quadratic_grad(self):
        np.random.seed(0)

        obj = Quadratic(
            A=jnp.asarray([[2.0, 1.0], [1.0, 5.0]]), b=jnp.asarray([1.0, 2.0])
        )

        for _ in range(10):
            x = np.random.randn(10, 2)
            np.testing.assert_allclose(obj.grad(x), self.exact_grad(obj, x), rtol=1e-5)


class LeastSquaresTest(absltest.TestCase):
    @staticmethod
    def exact_grad(obj: LeastSquares, x: jnp.ndarray) -> jnp.ndarray:
        """Returns exact gradient of the given objective at `x`."""
        n, d = obj.X.shape
        A = jnp.dot(obj.X.T, obj.X) / n + obj.lam * jnp.eye(d)
        b = jnp.dot(obj.X.T, obj.y) / n
        squeeze = False
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            squeeze = True
        value = vmap(lambda x: jnp.dot(A, x) - b)(x)
        if squeeze:
            value = jnp.squeeze(value)
        return value

    def test_least_squares_eval(self):
        np.random.seed(0)
        n, d = 100, 10
        batch_size = 10
        prng_seed = 0
        lam = 1e-3

        X = np.random.randn(n, d)
        y = X.dot(np.random.randn(d)) + np.random.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size, lam=lam)

        q_obj = Quadratic.from_least_squares(obj)
        np.testing.assert_allclose(obj.solve(), q_obj.solve())

        prng_key = random.PRNGKey(prng_seed)
        for _ in range(10):
            x = np.random.randn(np.random.randint(1, 5), d)
            prng_key, subkey = random.split(prng_key)
            obj_val_d = obj(x, prng_key=subkey, deterministic=True)
            np.testing.assert_allclose(obj_val_d, q_obj(x), rtol=1e-6)

    def test_least_squares_eval_stochastic(self):
        np.random.seed(0)
        n, d = 100, 10
        batch_size = 10
        prng_seed = 0
        lam = 1e-3

        X = np.random.randn(n, d)
        y = X.dot(np.random.randn(d)) + np.random.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size, lam=lam)

        prng_key = random.PRNGKey(prng_seed)

        subkey = random.split(prng_key, 1)[0]
        x_batch, y_batch = obj._sample_batch(batch_size, obj.data, subkey)
        obj_batch = LeastSquares(X=x_batch, y=y_batch, batch_size=batch_size, lam=lam)
        q_obj = Quadratic.from_least_squares(obj_batch)

        x = np.random.randn(d)
        obj_val_s = obj(x, prng_key=prng_key, deterministic=False)
        np.testing.assert_allclose(obj_val_s, q_obj(x), rtol=1e-6)

    def test_least_squares_grad(self):
        np.random.seed(0)
        n, d = 100, 10
        batch_size = 10
        prng_seed = 0
        lam = 1e-3

        X = np.random.randn(n, d)
        y = X.dot(np.random.randn(d)) + np.random.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size, lam=lam)
        q_obj = Quadratic.from_least_squares(obj)

        prng_key = random.PRNGKey(prng_seed)
        for _ in range(10):
            prng_key, subkey = random.split(prng_key)
            x = np.random.randn(np.random.randint(1, 5), d)
            obj_grad_d = obj.grad(x, prng_key=subkey, deterministic=True)
            np.testing.assert_allclose(obj_grad_d, q_obj.grad(x), rtol=1e-3)
            np.testing.assert_allclose(obj_grad_d, self.exact_grad(obj, x), rtol=1e-3)

    def test_least_squares_grad_stochastic(self):
        np.random.seed(0)
        n, d = 100, 10
        batch_size = 10
        prng_seed = 0
        lam = 1e-3

        X = np.random.randn(n, d)
        y = X.dot(np.random.randn(d)) + np.random.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size, lam=lam)

        prng_key = random.PRNGKey(prng_seed)

        subkey = random.split(prng_key, 1)[0]
        x_batch, y_batch = obj._sample_batch(batch_size, obj.data, subkey)
        obj_batch = LeastSquares(X=x_batch, y=y_batch, batch_size=batch_size, lam=lam)
        q_obj = Quadratic.from_least_squares(obj_batch)

        x = np.random.randn(d)
        obj_grad_s = obj.grad(x, prng_key=prng_key, deterministic=False)
        np.testing.assert_allclose(obj_grad_s, q_obj.grad(x), rtol=1e-5)


class CreateObjectives(absltest.TestCase):
    def test_create_quadratics(self):
        np.random.seed(0)
        objectives = create_random_quadratics(dim=10, num_objectives=5)

        self.assertTrue(len(objectives) == 5)
        for o in objectives:
            self.assertTrue(o.dim == 10)

        global_objective = create_global_quadratic(objectives, weights=np.ones(5))
        self.assertTrue(global_objective.dim == 10)

    def test_create_least_squares(self):
        np.random.seed(0)
        objectives = create_random_least_squares(
            num_objectives=5, batch_size=10, n_features=20
        )
        q_objectives = [Quadratic.from_least_squares(o) for o in objectives]

        self.assertTrue(len(objectives) == 5)
        for qo, lso in zip(q_objectives, objectives):
            self.assertTrue(qo.dim == lso.dim == 21)

        global_obj = create_global_least_squares(objectives)
        global_q_obj = create_global_quadratic(
            q_objectives, weights=np.asarray([o.num_points for o in objectives])
        )
        self.assertTrue(global_obj.dim == global_q_obj.dim == 21)
        np.testing.assert_allclose(global_obj.solve(), global_q_obj.solve(), rtol=1e-4)

        prng_key = random.PRNGKey(0)
        for _ in range(10):
            prng_key, subkey = random.split(prng_key)
            x = np.random.randn(np.random.randint(1, 5), global_obj.dim)
            obj_val_d = global_obj(x, prng_key=subkey, deterministic=True)
            np.testing.assert_allclose(obj_val_d, global_q_obj(x), rtol=1e-6)


if __name__ == "__main__":
    absltest.main()
