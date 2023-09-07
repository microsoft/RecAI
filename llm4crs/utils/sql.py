# import sqlparse
# from sqlparse.tokens import Text, Name, Literal


# def extract_comp_token(token):
#     if token.ttype is None:
#         for t in token.tokens:
#             res = extract_comp_token(t)
#             if t.ttype is None:
#                 res = extract_comp_token(t)
#             elif t.ttype == Name:   # column name
#                 res = {'col': t.value}
#             elif t.ttype == Literal.String.Single:
#                 res = {'entity': t.value}
#             else:
#                 res = {}
#     return 


# def parse_select(sql: str):
#     parsed = sqlparse.parse(sql)[0]
#     # fields
#     fields = []
#     for token in parsed.tokens:
#         if isinstance(token, sqlparse.sql.IdentifierList):
#             for identifier in token.get_identifiers():
#                 fields.append(identifier.value)
#         elif isinstance(token, sqlparse.sql.Identifier):
#             fields.append(token.value)

#     # tables
#     for token in parsed.tokens:
#         if isinstance(token, sqlparse.sql.Where):
#             table = token.tokens[2].value
#             break
#     else:
#         table = None

#     # conditions
#     for token in parsed.tokens:
#         if isinstance(token, sqlparse.sql.Where):
#             for t in token.tokens:
#                 if isinstance(t, sqlparse.sql.Comparison):

#             condition = token.tokens[4].value
#             break
#     else:
#         condition = None

#     return {'fields': fields, 'table': table, 'condition': condition}


# if __name__ == "__main__":
#     sql = r"SELECT name FROM games WHERE genre LIKE '%shooting%' AND price < 5;"
#     output = parse_select(sql)

#     print("End.")
