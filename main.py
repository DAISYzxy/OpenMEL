from maximum_spanning import *



# Test WikiMEL
with open('WikiMEL/WIKIMEL_test.json', 'r') as json_file:
    parsed_text = json.load(json_file)
    
with open('WikiMEL/mention_image_clip.json', 'r') as json_file:
    mention_clip = json.load(json_file)

with open('WikiMEL/entity_text_clip.json', 'r') as json_file:
    kb_text = json.load(json_file)

with open('WikiMEL/entity_image_clip.json', 'r') as json_file:
    kb_img = json.load(json_file)

for idx in range(len(kb_text)):
    kb_text[idx] = np.array(kb_text[idx])
    
for key in kb_img.keys():
    kb_img[key] = np.array(kb_img[key][0]).astype(np.float32)

with open('WikiMEL/mention_add_cap_clip.json', 'r') as json_file:
    Tm = json.load(json_file)

for idx in range(len(Tm)):
    Tm[idx] = np.array(Tm[idx][0])
Tm = np.array(Tm).astype(np.float32)

text_feature= []
for key in range(len(kb_text)):
    text_feature.append(kb_text[key][0])
text_feature = np.array(text_feature).astype(np.float32)

with open("WikiMEL/kb_entity.json", "r") as f:
    kb = json.load(f)
with open("WikiMEL/qid2id.json", "r") as f:
    qid2id = json.load(f)

K = 10
acc_1 = 0
acc_5 = 0
acc_10 = 0
dimension = text_feature.shape[1]
num_vectors = text_feature.shape[0]
faiss.normalize_L2(text_feature)
index = faiss.IndexFlatIP(dimension)
index.add(text_feature)
for idx in tqdm(range(len(parsed_text))):
    mention_text_query = Tm[idx]
    mention_img = parsed_text[idx]["imgPath"]
    threshold = 0.5
    match_found = False
    m = -1
    D_me = []
    I_me = []
    I_ee = []
    D_ee = []
    test_D_me = []
    test_I_me = []
    test_I_ee = []
    test_D_ee = []
    mention_img_query = 0
    if len(mention_img) > 1:
        mention_img_query = np.array(mention_clip[mention_img][0]).astype(np.float32)
        for key in range(len(kb)):
            entity_imgs = kb[key]["image_list"]
            img_path = ""
            if len(entity_imgs) > 0:
                img_path = entity_imgs[0]
            dot_product = np.dot(mention_text_query, text_feature[key])
            norm_query = np.linalg.norm(mention_text_query)
            norm_feature = np.linalg.norm(text_feature[key])
            tt_sim_me = dot_product / (norm_query * norm_feature)
            dot_product = np.dot(mention_img_query, text_feature[key])
            norm_query = np.linalg.norm(mention_img_query)
            norm_feature = np.linalg.norm(text_feature[key])
            vt_sim_me = dot_product / (norm_query * norm_feature)
            if img_path in kb_img.keys():
                dot_product = np.dot(mention_img_query, kb_img[img_path])
                norm_query = np.linalg.norm(mention_img_query)
                norm_feature = np.linalg.norm(kb_img[img_path])
                vv_sim_me = dot_product / (norm_query * norm_feature)
                dot_product = np.dot(mention_text_query, kb_img[img_path])
                norm_query = np.linalg.norm(mention_text_query)
                norm_feature = np.linalg.norm(kb_img[img_path])
                tv_sim_me = dot_product / (norm_query * norm_feature)
                max_sim_me = max(tt_sim_me, vt_sim_me, tv_sim_me)
                D_me.append(max_sim_me)
                I_me.append(key)
            else:
                max_sim_me = max(tt_sim_me, vt_sim_me)
                D_me.append(max_sim_me)
                I_me.append(key)
    else:
        for key in range(len(kb)):
            entity_imgs = kb[key]["image_list"]
            img_path = ""
            if len(entity_imgs) > 0:
                img_path = entity_imgs[0]
            dot_product = np.dot(mention_text_query, text_feature[key])
            norm_query = np.linalg.norm(mention_text_query)
            norm_feature = np.linalg.norm(text_feature[key])
            tt_sim_me = dot_product / (norm_query * norm_feature)
            if img_path in kb_img.keys():
                dot_product = np.dot(mention_text_query, kb_img[img_path])
                norm_query = np.linalg.norm(mention_text_query)
                norm_feature = np.linalg.norm(kb_img[img_path])
                tv_sim_me = dot_product / (norm_query * norm_feature)
                max_sim_me = max(tt_sim_me, tv_sim_me)
                D_me.append(max_sim_me)
                I_me.append(key)
            else:
                D_me.append(tt_sim_me)
                I_me.append(key)
    
    # For WikiDiverse, we do not require additional filtering by mention names as discussed in our paper.
    # For WikiMEL and RichpediaMEL datasets, please direct use the complete codes.
    for k in range(len(kb)):
        sim = similar_strings(parsed_text[idx]["mentions"], kb[k]["entity_name"])
        if sim >= threshold:
            test_D_me.append(D_me[k])
            test_I_me.append(k)
            match_found = True
    
    
    if match_found:
        combined_list = list(zip(test_D_me, test_I_me))
        combined_list.sort(reverse=True, key=lambda x: x[0])
        test_D_me_sorted, test_I_me_sorted = zip(*combined_list)
        test_D_me = list(test_D_me_sorted)
        test_I_me = list(test_I_me_sorted)
        
    if not match_found:
        combined_list = list(zip(D_me, I_me))
        combined_list.sort(reverse=True, key=lambda x: x[0])
        test_D_me_sorted, test_I_me_sorted = zip(*combined_list)
        test_D_me = list(test_D_me_sorted)[:K]
        test_I_me = list(test_I_me_sorted)[:K]
        entity_text_query0 = text_feature[test_I_me]
        faiss.normalize_L2(entity_text_query0)
        D_ee, I_ee = index.search(entity_text_query0, 5)
        for idx in range(len(test_I_me)):
            e_idx = test_I_me[idx]
            test_D_ee[e_idx] = np.array([item * test_D_me[idx] for item in D_ee[idx]])
            test_I_ee[e_idx] = np.array(I_ee[idx])


    res, mst = maximum_spanning_tree(m, test_I_me, test_D_me, test_I_ee, test_D_ee, 21)

    # Obtain the Top B Results as res[1:B+1] (res[0] is the mention node denoted as -1)
    qidres = []
    for ridx in res[1:2]: # Top 1
        qidres.append(kb[ridx]["qid"])
        
    if parsed_text[idx]["answer"] in qidres:
        acc_1 += 1

    qidres = []
    for ridx in res[1:6]: # Top 5
        qidres.append(kb[ridx]["qid"])
        
    if parsed_text[idx]["answer"] in qidres:
        acc_5 += 1

    qidres = []
    for ridx in res[1:11]: # Top 10
        qidres.append(kb[ridx]["qid"])
        
    if parsed_text[idx]["answer"] in qidres:
        acc_10 += 1
        

print(f"HIT@1: {acc_1 / len(parsed_text)}")
print(f"HIT@5: {acc_5 / len(parsed_text)}")
print(f"HIT@10: {acc_10 / len(parsed_text)}")




