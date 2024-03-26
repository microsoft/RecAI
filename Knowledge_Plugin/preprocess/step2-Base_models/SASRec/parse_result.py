in_file = "/home/wei_xu/rec/SASRec/steam_default/log.txt"

def all_number(line):
    cnt = 0
    for x in line.strip().split():
        try:
            cnt += 1
            number = float(x)
        except:
            return False
    if cnt:
        return True
    return False

ndcg = {}
rec = {}
map = {}

for line in open(in_file, "r"):
    if "Evaluating and Testing..." in line:
        epoch = line.split("Evaluating and Testing...")[0]
    if "validating..." in line:
        valid_or_test = "Valid"
    if "testing..." in line:
        valid_or_test = "Test"
    if "ranking" in line:
        rank_or_retrieval = "rank"
    if "retrieval" in line:
        rank_or_retrieval = "retrieval"
    if "@" in line:
        topk = int(line[line.find("@")+1:line.find("@")+4])
    if all_number(line):
        l = []
        for x in line.strip().split():
            number = float(x)
            l.append(number)
        ndcg[topk] = l[0]
        rec[topk] = l[1]
        map[topk] = l[4]
        if topk == 100:
            print(f"| epoch:{epoch} | {valid_or_test} | {rank_or_retrieval} |")
            print(f"{ndcg[1]}\t{ndcg[5]}\t{ndcg[10]}\t{ndcg[20]}\t{ndcg[100]}\t" +
                  f"{rec[1]}\t{rec[5]}\t{rec[10]}\t{rec[20]}\t{rec[100]}\t" +
                  f"{map[1]}\t{map[5]}\t{map[10]}\t{map[20]}\t{map[100]}\t")