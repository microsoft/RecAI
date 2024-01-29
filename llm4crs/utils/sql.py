# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sqlparse

def extract_columns_from_where(sql_query):
    # Parse the SQL statement using sqlparse
    parsed = sqlparse.parse(sql_query)

    # Iterate through the parsed tokens and find the WHERE clause
    where_conditions = []
    for statement in parsed:
        for token in statement.tokens:
            if isinstance(token, sqlparse.sql.Where):
                # Extract conditions from the WHERE clause
                where_conditions.extend(token.tokens)

    # Iterate through the conditions in the WHERE clause and extract column names
    column_names = set()
    for condition in where_conditions:
        if isinstance(condition, sqlparse.sql.Comparison):
            # Extract identifiers from both sides of the comparison operator
            for part in condition.tokens:
                if isinstance(part, sqlparse.sql.Identifier):
                    for subpart in part.flatten():
                        if subpart.ttype == sqlparse.tokens.Name:
                            column_names.add(subpart.value)

    return list(column_names)
