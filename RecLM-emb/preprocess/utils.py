# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from collections import defaultdict
from tqdm import tqdm
import random
import pandas as pd
import math
import string
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar 
  
def get_price_date_stats(itemid2price_date_map):
    max_price = max([x['price'] for x in itemid2price_date_map[1:] if x['price'] is not None])
    min_price = min([x['price'] for x in itemid2price_date_map[1:] if x['price'] is not None])
    max_date, min_date = None, None
    for x in itemid2price_date_map[1:]:
        if x['release date'] is not None:
            if max_date is None or (x['release date'].year, x['release date'].month) > (max_date.year, max_date.month):
                max_date = x['release date']
            if min_date is None or (x['release date'].year, x['release date'].month) < (min_date.year, min_date.month):
                min_date = x['release date']
    return {'max_price': max_price, 'min_price': min_price, 'max_date': max_date, 'min_date': min_date}

def transform_date_format(date_string):  
    try:  
        # Try to parse the string with format "%Y-%m-%d"  
        date = datetime.strptime(date_string, "%Y-%m-%d")
        return date
    except ValueError:  
        pass  
  
    try:  
        # Try to parse the string with format "%B %d, %Y"  
        date = datetime.strptime(date_string, "%B %d, %Y")  
        return date
    except ValueError:  
        pass  
    return None

def load_titleid_2_index(infile):
    data = pd.read_json(infile, lines=True)
    res = {}
    n = len(data)
    for i in range(n):
        row = data.iloc[i]
        res[str(row['TitleId'])] = int(row['HashId'])
    return res

def get_value_by_key(keys, d):
    for key in keys:
        if key in d:
            return d[key]
    return None

def get_item_stats(in_seq_data, in_meta_data):
    total_num = 0
    item2date = {}
    item2price = {}
    for idx, line in enumerate(open(in_meta_data)):
        total_num += 1
        line = json.loads(line)
        release_date = get_value_by_key(['ReleaseDateCurated', 'release_date'], line)
        price = get_value_by_key(['SuggestPrice', 'price'], line)
        if release_date:
            date_object = transform_date_format(release_date)
            if date_object:
                item2date[idx+1] = date_object
        if price:
            try:
                item2price[idx+1] = float(price)
            except:
                item2price[idx+1] = 0.0
            
    
    threshold = math.floor(total_num*0.05)
    sorted_item2date = sorted(item2date.items(), key=lambda x: x[1], reverse=True)
    sorted_item2price = sorted(item2price.items(), key=lambda x: x[1], reverse=True)

    recent_itemset = set([x[0] for x in sorted_item2date[:threshold]])
    expensive_itemset = set([x[0] for x in sorted_item2price[:threshold]])
    cheap_itemset = set([x[0] for x in sorted_item2price[-threshold:]])

    item2freq = defaultdict(int)
    for idx, line in tqdm(enumerate(open(in_seq_data)), desc='item freq'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')[:-1] # remove the last item
        for itemid in itemids:
            item2freq[int(itemid)] += 1
    sorted_item2freq = sorted(item2freq.items(), key=lambda x: x[1], reverse=True)
    popular_itemset = set([x[0] for x in sorted_item2freq[:threshold]])

    return recent_itemset, cheap_itemset, expensive_itemset, popular_itemset, total_num

def get_feature2itemid(in_meta_data):
    tags2itemid = defaultdict(set)
    game_details2itemid = defaultdict(set)
    publisher2itemid = defaultdict(set)
    developer2itemid = defaultdict(set)
    primary_genre2itemid = defaultdict(set)
    subgenre2itemid = defaultdict(set)
    for idx, line in enumerate(open(in_meta_data)):
        line = json.loads(line)

        tags = []
        if 'GPT_tags' in line:
            tags = line['GPT_tags'].split(',')
        elif 'tags' in line:
            tags = line['tags']
        elif 'popular_tags' in line:
            tags = line['popular_tags']
        elif 'genres' in line:
            tags = line['genres']

        game_details = []
        if 'specs' in line and len("".join(line['specs']))>0:
            game_details = line['specs']
        elif 'game_details' in line and len(line['game_details'])>0:
            game_details = line['game_details'].split(',')

        publisher = get_value_by_key(['PublisherCurated', 'publisher'], line)
        developer = get_value_by_key(['DeveloperCurated', 'developer'], line)
        primary_genre = get_value_by_key(['PrimaryGenre'], line)
        subgenre = get_value_by_key(['SubGenre'], line)

        if len(tags)>0:
            for tag in tags:
                tags2itemid[tag].add(idx+1)
        if len(game_details)>0:
            for game_detail in game_details:
                game_details2itemid[game_detail].add(idx+1)
        if publisher:
            publisher2itemid[publisher].add(idx+1)
        if developer:
            developer2itemid[developer].add(idx+1)
        if primary_genre:
            primary_genre2itemid[primary_genre].add(idx+1)
        if subgenre:
            subgenre2itemid[subgenre].add(idx+1)
    output = {}
    if len(tags2itemid)>0:
        output['tags: '] = tags2itemid
    if len(game_details2itemid)>0:
        output['game details: '] = game_details2itemid
    if len(publisher2itemid)>0:
        output['publisher: '] = publisher2itemid
    if len(developer2itemid)>0:
        output['developer: '] = developer2itemid
    if len(primary_genre2itemid)>0:
        output['primary genre: '] = primary_genre2itemid
    if len(subgenre2itemid)>0:
        output['subgenre: '] = subgenre2itemid
    return output

def get_item_text(infile_metadata, save_item_prompt_path=None):
    itemid2title = [('padding', None)]
    itemid2text = ['padding']
    itemid2features = [('padding', None)]
    itemid2price_date_map = [{'price': None, 'release date': None, 'next month': None, 'last month': None}]
    for idx, line in enumerate(open(infile_metadata)):
        line = json.loads(line)

        ## TitleName is Xbox data format; app_name and title are Steam data format
        title = get_value_by_key(['TitleName', 'app_name', 'title'], line) 
        if not title:
            raise Exception("No title in line: {}".format(line))

        tags = []
        if 'GPT_tags' in line:
            tags = line['GPT_tags'].split(',')
        elif 'tags' in line:
            tags = line['tags']
        elif 'popular_tags' in line:
            tags = line['popular_tags']
        elif 'genres' in line:
            tags = line['genres']
         

        game_details = []
        if 'specs' in line and len("".join(line['specs']))>0:
            game_details = line['specs']
        elif 'game_details' in line and len(line['game_details'])>0:
            game_details = line['game_details'].split(',')

        publisher = get_value_by_key(['PublisherCurated', 'publisher'], line)
        developer = get_value_by_key(['DeveloperCurated', 'developer'], line)
        primary_genre = get_value_by_key(['PrimaryGenre'], line)
        subgenre = get_value_by_key(['SubGenre'], line)
        price = get_value_by_key(['SuggestPrice', 'price'], line)
        release_date = get_value_by_key(['ReleaseDateCurated', 'release_date'], line)

        description = ""
        if "ProductDescription" in line:
            description = line["ProductDescription"].strip()
        elif "desc_snippet" in line and len(line["desc_snippet"].strip())>0:
            description = line["desc_snippet"].strip()
        elif "game_description" in line and len(line["game_description"].strip())>0:
            description = line["game_description"].strip()
        elif "description" in line and len(line["description"].strip())>0:
            description = line["description"].strip()
        description = description.strip(string.punctuation+string.whitespace)
    
        prompt = ""
        prompt += 'title: ' + title + ", "
        itemid2title.append(('title: ', title))
        itemid2features.append([])
        itemid2price_date_map.append({'price': None, 'release date': None, 'next month': None, 'last month': None})
        if len(tags)>0:
            prompt += 'tags: ' + ",".join(tags) + ", "
            itemid2features[-1].append(('tags: ', tags))
        if len(game_details)>0:
            prompt += 'game details: ' + ",".join(game_details) + ", "
            itemid2features[-1].append(('game details: ', game_details))
        if publisher:
            prompt += 'publisher: ' + publisher + ", "
            itemid2features[-1].append(('publisher: ', publisher))
        if developer:
            prompt += 'developer: ' + developer + ", "
            itemid2features[-1].append(('developer: ', developer))
        if primary_genre:
            prompt += 'primary genre: ' + primary_genre + ", "
            itemid2features[-1].append(('primary genre: ', primary_genre))
        if subgenre:
            prompt += 'subgenre: ' + subgenre + ", "
            itemid2features[-1].append(('subgenre: ', subgenre))
        if price:
            try:
                price = float(price)
            except:
                price = 0.0
            prompt += 'price: ' + str(price) + ", "
            itemid2features[-1].append(('price: ', str(price)))
            itemid2price_date_map[-1]['price'] = price
        if release_date:
            date_object = transform_date_format(release_date)
            if date_object:
                date_string = date_object.strftime("%B %d, %Y")
                prompt += 'release date: ' + date_string + ", "
                itemid2features[-1].append(('release date: ', date_string))
                itemid2price_date_map[-1]['release date'] = date_object
                itemid2price_date_map[-1]['next month'] = itemid2price_date_map[-1]['release date'] + relativedelta(months=+1)
                itemid2price_date_map[-1]['last month'] = itemid2price_date_map[-1]['release date'] + relativedelta(months=-1)
        if len(description)>0:
            prompt += 'description: ' + description
            itemid2features[-1].append(('description: ', description))
        prompt = prompt.strip()
        prompt = prompt.strip(',')
        itemid2text.append(prompt)
    
    if save_item_prompt_path!=None:
        with open(save_item_prompt_path, 'w', encoding='utf-8') as f:
            for id, prompt in enumerate(itemid2text):
                line = {'id': id, 'text': prompt, 'title': itemid2title[id], 'features': itemid2features[id], 'price': itemid2price_date_map[id]['price'],
                        'release date': None, 'next month': None, 'last month': None}
                if itemid2price_date_map[id]['release date']!=None:
                        line['release date'] = itemid2price_date_map[id]['release date'].strftime("%B %d, %Y")
                        line['next month'] = itemid2price_date_map[id]['next month'].strftime("%B %d, %Y")
                        line['last month'] = itemid2price_date_map[id]['last month'].strftime("%B %d, %Y")
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

    return itemid2text, itemid2title, itemid2features, itemid2price_date_map


def text4query2item(target_features, target_item_title, min_features, max_features, min_tags, max_tags):
    query = ''
    has_prefix = False if random.random() < 0.5 else True
    sampled_features = random.sample(target_features, random.randint(min_features, max_features))
    ground_truth = []
    for key, value in sampled_features:
        if 'game details: '==key or 'tags: '==key:
            features_value = random.sample(value, random.randint(min(min_tags, len(value)), min(max_tags, len(value))))
            ground_truth.append((key, features_value))
                
            if has_prefix:
                query += key + ','.join(features_value) + ', '
            else:
                query += ','.join(features_value) + ', '
        elif 'description: '==key:
            features_value = value.replace(target_item_title, 'this game')
            ground_truth.append((key, value))
            if has_prefix:
                query += key + features_value + ', '
            else:
                query += features_value + ', '
        else:
            ground_truth.append((key, value))
            if has_prefix:
                query += key + value + ', '
            else:
                query += value + ', '
    query = query.strip().strip(',')
    return query, sampled_features, ground_truth

def cal_item2pos(in_seq_data):
    item2item_freq = defaultdict(int)
    item_degree = defaultdict(int)
    for idx, line in tqdm(enumerate(open(in_seq_data)), desc='item freq & degree'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')[:-1] # remove the last item
        for itemid in itemids:
            item_degree[int(itemid)] += len(itemids)-1
        for i in range(len(itemids)-1):
            for j in range(i+1, len(itemids)):
                item2item_freq[(int(itemids[i]), int(itemids[j]))] += 1
                item2item_freq[(int(itemids[j]), int(itemids[i]))] += 1
        # if idx>10000:
        #     break

    item2item_sim = defaultdict(list)
    for item_pair, freq in tqdm(item2item_freq.items(), desc='filtering', total=len(item2item_freq)):
        sim = freq / (math.sqrt(item_degree[item_pair[0]] * item_degree[item_pair[1]]) + 10)
        item2item_sim[item_pair[0]].append((item_pair[1], sim))

    item2pos = defaultdict(set)
    for item, sim_list in item2item_sim.items():
        sim_list.sort(key=lambda x: x[1], reverse=True)
        pos_list = [x[0] for x in sim_list[:20]]
        item2pos[item] = set(pos_list)
        # item2pos[item_pair[0]].add(item_pair[1])
        # item2pos[item_pair[1]].add(item_pair[0])
    return item2pos

def text4item2item(features, item_title):
    query = ''
    has_prefix = False if random.random() < 0.5 else True
    # if random.random() < 0.5:
    #     template = "{}"
    # else:
    
    if has_prefix:
        query += 'title: ' + item_title + ', '
    else:
        query += item_title + ', '
    if random.random() < 0.5: #more features than title
        sampled_features = random.sample(features, random.randint(1, len(features)))
        for key, value in sampled_features:
            if 'game details: '==key or 'tags: '==key:
                features_value = ','.join(random.sample(value, random.randint(1, len(value))))
                if has_prefix:
                    query += key + features_value + ', '
                else:
                    query += features_value + ', '
            else:
                if has_prefix:
                    query += key + value + ', '
                else:
                    query += value + ', '
        
    query = query.strip().strip(',')
    return query

def random_replace(game_name, probability=0.1):  
    game_name_list = list(game_name)
    new_game_name = []
    for i in range(len(game_name_list)):  
        if random.random() < probability and game_name_list[i] not in string.whitespace:
            if random.random() < 0.5: 
                new_game_name.append(random.choice(string.ascii_letters+string.digits+string.punctuation))
            else:
                pass
        else:
            new_game_name.append(game_name_list[i])
    return ''.join(new_game_name)

def vaguequery(price, date, next_month, last_month, price_date_stats):
    # if price < math.ceil(price_date_stats['min_price'])+10 or (price < math.floor(price_date_stats['max_price'])-10 and random.random() < 0.5): #less than
    #     price_noise = random.randint(math.ceil(price), math.floor(price_date_stats['max_price']))
    #     price_flag = ('less than', float(price_noise))
    # else:
    #     price_noise = random.randint(math.ceil(price_date_stats['min_price']), math.floor(price))
    #     price_flag = ('more than', float(price_noise))

    # for steam dataset, the distribution of price is not uniform, so we use the following method to sample price
    if price < math.ceil(price_date_stats['min_price'])+0.5 or (price < math.floor(price_date_stats['max_price'])-10 and random.random() < 0.5): #less than
        if price < 35.0 and random.random() < 0.8:
            price_max = 35
        else:
            price_max = math.floor(price_date_stats['max_price'])
        price_noise = random.randint(math.ceil(price), price_max)
        price_flag = ('less than', float(price_noise))
    else:
        price_noise = random.randint(math.ceil(price_date_stats['min_price']), math.floor(price))
        price_flag = ('more than', float(price_noise))
    price_query = 'price '+price_flag[0]+' '+str(price_flag[1])
    
    year_choices = ['before', 'after', 'in']
    if date.year >= price_date_stats['max_date'].year:
        year_choices.remove('before')
    if date.year <= price_date_stats['min_date'].year:
        year_choices.remove('after')
    year_flag = random.choice(year_choices)
    if random.random() <= 0.4:# only year
        month_flag = 0
        if year_flag == 'before':
            year_flag = ('before', random.randint(date.year+1, price_date_stats['max_date'].year))
        elif year_flag == 'after':
            year_flag = ('after', random.randint(price_date_stats['min_date'].year, date.year-1))
        else:
            year_flag = ('in', date.year)
        date_query = year_flag[0]+' '+str(year_flag[1])  
    else:
        month_flag = 1
        if year_flag == 'before':
            month_noise = random.randint(0, (price_date_stats['max_date'].year-date.year)*12 - date.month)
            year_flag = ('before', next_month + relativedelta(months=+month_noise))
        elif year_flag == 'after':
            month_noise = random.randint(0, (date.year-price_date_stats['min_date'].year)*12 - price_date_stats['min_date'].month)
            year_flag = ('after', last_month + relativedelta(months=-month_noise))
        else:
            year_flag = ('in', date)
        date_query = year_flag[0]+' '+str(calendar.month_name[year_flag[1].month])+', '+str(year_flag[1].year)
    date_query = 'release date '+date_query

    combine_flag = random.choice([0, 1, 2, 3])# 0: price, 1: date, 2: price+date, 3: date+price
    if combine_flag == 0:
        query = price_query
    elif combine_flag == 1:
        query = date_query
    elif combine_flag == 2:
        query = price_query+', '+date_query
    elif combine_flag == 3:
        query = date_query+', '+price_query
    
    return query, combine_flag, price_flag, month_flag, year_flag

def vaguequery_neg_sample(combine_flag, price_flag, month_flag, year_flag, itemid2price_date_map, args):
    neg_items = []
    max_retries = args.neg_num*3
    cur_retries = 0
    while len(neg_items) < args.neg_num and cur_retries < max_retries:
        cur_retries += 1
        neg_item = random.randint(1, len(itemid2price_date_map)-1)
        if combine_flag in [0, 2, 3]:#check price
            if itemid2price_date_map[neg_item]['price']==None:
                neg_items.append(neg_item)
                continue
            if price_flag[0] == 'less than' and price_flag[1] <= itemid2price_date_map[neg_item]['price']:
                    neg_items.append(neg_item)
                    continue
            elif price_flag[0] == 'more than' and price_flag[1] >= itemid2price_date_map[neg_item]['price']:
                    neg_items.append(neg_item)
                    continue
        
        if combine_flag in [1, 2, 3]:#check date
            if itemid2price_date_map[neg_item]['release date']==None:
                neg_items.append(neg_item)
                continue
            if year_flag[0]=='before':
                if not month_flag:
                    if year_flag[1] <= itemid2price_date_map[neg_item]['release date'].year:
                        neg_items.append(neg_item)
                        continue
                elif (year_flag[1].year, year_flag[1].month)<=(itemid2price_date_map[neg_item]['release date'].year, itemid2price_date_map[neg_item]['release date'].month):
                    neg_items.append(neg_item)
                    continue
            elif year_flag[0]=='after':
                if not month_flag:
                    if year_flag[1] >= itemid2price_date_map[neg_item]['release date'].year:
                        neg_items.append(neg_item)
                        continue
                elif (year_flag[1].year, year_flag[1].month)>=(itemid2price_date_map[neg_item]['release date'].year, itemid2price_date_map[neg_item]['release date'].month):
                    neg_items.append(neg_item)
                    continue
            elif year_flag[0]=='in':
                if not month_flag:
                    if year_flag[1] != itemid2price_date_map[neg_item]['release date'].year:
                        neg_items.append(neg_item)
                        continue
                elif year_flag[1].year!=itemid2price_date_map[neg_item]['release date'].year or year_flag[1].month != itemid2price_date_map[neg_item]['release date'].month:
                    neg_items.append(neg_item)
                    continue
    
    # if len(neg_items) < args.neg_num:
    #     neg_items.extend(random.sample(range(1, len(itemid2price_date_map)-1), args.neg_num-len(neg_items)))

    return neg_items

def text4negquery(sample_names_l1, sample_names_l2, itemid2text, features2itemids):
    list_names_l1 = [x for x in ['tags: ', 'game details: '] if x in sample_names_l1]
    if len(list_names_l1)>0 and random.random() < 0.2:
        names_l1 = random.sample(list_names_l1, 1)
    else:
        names_l1 = random.sample(sample_names_l1, random.randint(1, 2))
        
    names_l2 = {x: random.sample(sample_names_l2[x], random.randint(1, 3)) if x in ['tags: ', 'game details: '] else random.sample(sample_names_l2[x], 1) for x in names_l1}
    query = ''
    pos_set = set(range(1, len(itemid2text)))
    neg_set = set()
    for key, value in names_l2.items():
        query += key + ','.join(value) + ', '
        for v in value:
            neg_set.update(features2itemids[key][v])
            pos_set.difference_update(features2itemids[key][v])
    query = query.strip().strip(',')
    return query, pos_set, neg_set