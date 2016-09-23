from __future__ import absolute_import
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from .utils import add_pyspark_path_if_needed, quiet_py4j

add_pyspark_path_if_needed()

from .testcase import SparkTestingBaseReuse

import os
import sys
from itertools import chain
import time
import operator
import tempfile
import random
import struct
from functools import reduce

from pyspark.context import SparkConf, SparkContext, RDD
from pyspark.sql import SQLContext


class SQLTestCase(SparkTestingBaseReuse):

    """Basic common test case for Spark SQL tests. Provides a
    Spark SQL context as well as some helper methods for comparing
    results."""

    @classmethod
    def setUpClass(cls):
        super(SQLTestCase, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(SQLTestCase, cls).tearDownClass()

    def setUp(self):
        self.sqlCtx = SQLContext(self.sc)

    def assertSchemaEqual(self, expected, result):
        """
        Make sure that schema is as we expect it

        :param expected: a schema, can be found in a dataframe with dataframe_obj.schema
        :param result: as above
        """
        self.assertEqual(len(expected.simpleString()), len(result.simpleString()))
        expected_lookup = dict((x.name, x) for x in vars(expected)['fields'])

        for field in vars(result)['fields']:
            wanted = expected_lookup.get(field.name)

            self.assertIsNotNone(wanted,
                    '{0} is not in schema'.format(field.name))
            self.assertEqual(type(wanted.dataType),
                    type(field.dataType),
                    'Field {0} is not the expected type'.format(field.name))
            self.assertEqual(wanted.nullable,
                    field.nullable,
                    'Nullable property not as expected for {0}'.format(field.name))

    def assertSchemaEqualIgnoringNullable(self, expected, result):
        """
        Same with assertSchemaEqual but ignoring Nullable property of any field.
        """
        fields_x = vars(expected).get('fields')
        fields_y = vars(result).get('fields')
        self.assertEqual(len(fields_x), len(fields_y))

        expected_lookup = dict((x.name, x) for x in fields_x)

        for field in fields_y:
            wanted = expected_lookup.get(field.name)

            self.assertIsNotNone(wanted,
                                 '{0} is not in schema'.format(field.name))
            self.assertEqual(type(wanted.dataType),
                             type(field.dataType),
                             'Field {0} is not the expected type'.format(field.name))

    def assertDataFrameEqual(self, expected, result, tol=0):
        """Assert that two DataFrames contain the same data.
        When comparing inexact fields uses tol.
        """
        self.assertSchemaEqualIgnoringNullable(expected.schema, result.schema)
        try:
            expectedRDD = expected.rdd.cache()
            resultRDD = result.rdd.cache()

            def zipWithIndex(rdd):
                """Zip with index (idx, data)"""
                return rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

            def equal(x, y):
                if len(x) != len(y):
                    return False
                elif x == y:
                    return True
                else:
                    dict_x = x.asDict()
                    dict_y = y.asDict()
                    if dict_x == dict_y:
                        return True
                    elif tol > 0:
                        for key in dict_x.keys():
                            a = dict_x.get(key)
                            b = dict_y.get(key)
                            if isinstance(a, float) and isinstance(b, float):
                                if abs(a - b) > tol:
                                    return False
                            else:
                                if a != b:
                                    return False
                return True
            expectedIndexed = zipWithIndex(expectedRDD)
            resultIndexed = zipWithIndex(resultRDD)
            joinedRDD = expectedIndexed.join(resultIndexed)
            unequalRDD = joinedRDD.filter(
                lambda x: not equal(x[1][0], x[1][1]))
            differentRows = unequalRDD.take(10)
            self.assertEqual([], differentRows)
        finally:
            expectedRDD.unpersist()
            resultRDD.unpersist()
