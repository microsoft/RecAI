# python gen_data.py --dataset ml1m --method ICL
# python gen_data.py --dataset beauty --method ICL
# python gen_data.py --dataset online_retail --method ICL

# python gen_answer.py --dataset ml1m --method ICL
# python gen_answer.py --dataset beauty --method ICL
# python gen_answer.py --dataset online_retail --method ICL

python gen_data.py --dataset online_retail --method preference
python gen_answer.py --dataset online_retail --method preference
python eval.py --dataset online_retail --method preference
# python eval.py --dataset online_retail --method preference
# python eval.py --dataset beauty --method preference
