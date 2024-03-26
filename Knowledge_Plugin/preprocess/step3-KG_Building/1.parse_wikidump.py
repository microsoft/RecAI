# process the knowledge graph wikidata

import os
import json
import gzip
import argparse
from tqdm import tqdm
from multiprocessing import Process, JoinableQueue, Lock

def producer(queue:JoinableQueue, dump_path):
    fr = gzip.open(dump_path, "rb")
    fr.readline()
    for line in tqdm(fr):
        queue.put(line)

def process_line(work_id, queue, lock, args):
    description_f = open(os.path.join(args.output_path, "description.txt"), "a", encoding="utf-8")
    label_f = open(os.path.join(args.output_path, "label.txt"), "a", encoding="utf-8")
    triple_f = open(os.path.join(args.output_path, "triple.txt"), "a", encoding="utf-8")

    while True:
        try:
            line = queue.get()
            line = json.loads(line.rstrip(b",\n"))
            id = line["id"]
            if "en" in line["descriptions"]:
                description = line["descriptions"]["en"]["value"]
            else:
                description = "no description"
            if "en" in line["labels"]:
                title = line["labels"]["en"]["value"]
            else:
                title = "no title"
            if "en" in line["aliases"]:
                aliases = [x["value"] for x in line["aliases"]["en"]]
            else:
                aliases = []
            label = [title] + aliases
            description = description.replace('\r',' ').replace('\n',' ').replace('\t',' ')
            label = [x.replace('\r',' ').replace('\n',' ').replace('\t',' ') for x in label]
            triples = []
            if line["type"] == "item":
                for _, targets in line["claims"].items():
                    for meta in targets:
                        meta = meta["mainsnak"]
                        if "datatype" in meta and meta["datatype"] == "wikibase-item" and "datavalue" in meta and meta["datavalue"]["value"]["entity-type"] == "item":
                            triples.append((id, meta["property"], meta["datavalue"]["value"]["id"]))
            lock.acquire()
            description_f.write(f"{id}\t{description}\n")
            description_f.flush()
            label = '\t'.join(label)
            label_f.write(f"{id}\t{label}\n")
            label_f.flush()
            if triples:
                for triple in triples:
                    triple_f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
                triple_f.flush()
            lock.release()
        except:
            pass
        finally:
            queue.task_done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', type=str, default="../../KG_data/latest-all.json.gz")
    parser.add_argument('--output_path', type=str, default="data/KG_data")
    parser.add_argument('--num_works', type=int, default=18)
    args = parser.parse_args()
    
    description_f = open(os.path.join(args.output_path, "description.txt"), "w", encoding="utf-8")
    label_f = open(os.path.join(args.output_path, "label.txt"), "w", encoding="utf-8")
    triple_f = open(os.path.join(args.output_path, "triple.txt"), "w", encoding="utf-8")
    description_f.close()
    label_f.close()
    triple_f.close()

    write_lock = Lock()
    
    queue = JoinableQueue(100)
    pc = Process(target=producer, args=(queue,args.dump_path))
    pc.start()

    workercount = args.num_works
    for i in range(workercount):
        worker = Process(target=process_line, args=(i, queue, write_lock, args,))
        worker.daemon = True
        worker.start()
    pc.join()
    queue.join()