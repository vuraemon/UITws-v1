import copy # for copying something in Python
import unicodedata
import multiprocessing as mp
from multiprocessing import Manager
import gc
import numpy as np

class WSUtils():
    def __init__(self, VNDict):
        ##################################################################################################
        '''
        The dictionary bellow is normalizing map which is inherited from Dat Quoc Nguyen
        (https://github.com/datquocnguyen/RDRsegmenter)
        (https://github.com/datquocnguyen/VnMarMoT/blob/master/Utility.py)
        '''
        self.normalize_map = {
            "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã",
            "ọa": "oạ", "òe": "oè", "óe": "oé", "ỏe": "oẻ",
            "õe": "oẽ", "ọe": "oẹ", "ùy": "uỳ", "úy": "uý",
            "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ", "Ủy": "Uỷ"
        }
        ##################################################################################################
        '''
        The RDR_VNDict, VNFamilyName, and VNMiddle are inherited from the work of Dat Quoc Nguyen
        (https://github.com/datquocnguyen/RDRsegmenter)
        The UET_VNDict is inherited from the work of Tuan-Phong Nguyen
        (https://github.com/phongnt570/UETsegmenter/blob/master/dictionary/VNDictObject)
        '''
        ##################################################################################################
        self.VNDict       = set(self.read_lines('./dict/' + VNDict + '.txt'))
        self.VNFamilyName = set(self.read_lines('./dict/VNFamilyName.txt'))
        self.VNMiddle     = set(self.read_lines('./dict/VNMiddle.txt'))
        ##################################################################################################
        self.lower        = '🄰'  # this string indicates for "lower" token style
        self.upper        = '🄱'  # this string indicates for "upper" token style
        self.bos          = '🄲'  # this string indicates for "begin of sentence" token style
        self.eos          = '🄳'  # this string indicates for "end of sentence" token style
        self.allupper     = '🄴'  # this string indicates for "all characters are upper" token style
        self.other        = '🄵'  # this string indicates for "other" token style (not in of all style above)
        ##################################################################################################      
        
    
    def pop_seen_words_sfx(self, train_stns, ratios):
        seen_words     = set()
        seen_words_sfx = set()

        for stn in train_stns:
            wrds = stn.lower().split(' ')
            for w in wrds:
                seen_words.update({w})
                
                syl = w.split('_')
                sfx = syl[-1]

                if len(syl) in [3, 4] and sfx in ratios['sfx'] and not self.inVNFamilyName(syl[0]):
                    seen_words_sfx.update({w})
                    
        seen_words.update(self.VNDict)
        
        
        for word in seen_words_sfx:
            if self.inVNDict(word.replace('_', ' ')):
                self.VNDict.discard(word.replace('_', ' '))

        return seen_words, seen_words_sfx
    

    def get_unseenwords_sfx(self, test_stns, seen_words, ratios):
        unseen_words_sfx = set()

        for stn in test_stns:
            wrds = stn.lower().split(' ')
            for w in wrds:
                syl = w.split('_')
                sfx = syl[-1]

                if len(syl) in [3, 4] and sfx in ratios['sfx'] and not self.inVNFamilyName(syl[0]):
                    if ' '.join(syl) not in seen_words:
                        unseen_words_sfx.update({w})

        return unseen_words_sfx
       

    def normalize_accent(self, line_pos):
        for key in self.normalize_map.keys():
            line_pos = line_pos.replace(key, self.normalize_map[key])
        return unicodedata.normalize('NFC', u'' + line_pos)
    
    def possible_tag(self, stns_train, POSs):
        possible_lbl = {}
        
        for idx, line in enumerate(stns_train):
            lbl = POSs[idx]
            for j, word in enumerate(line):
                word_lower = word.lower()
                if word not in possible_lbl:
                    possible_lbl[word_lower] = [lbl[j]]
                else:
                    if lbl[j] not in possible_lbl[word_lower]:
                        possible_lbl[word_lower].append(lbl[j])
        
        return possible_lbl
    
    
    def read_lines(self, file_path):
        '''
        Input: The path of the text file.
        Output: The list in which each line of the list according to each line in the input text file
        '''
        lines = []
        f = open(file_path, 'r', encoding='utf8')
        for line in f.readlines():
            line_pos = line.replace('\n', '')
            lines.append(self.normalize_accent(line_pos))
        f.close()

        return lines
    
    
    def read_ws_corpus(self, file_path):
        '''
        Input: The path of the text file.
        Output: The list in which each line of the list according to each line in the input text file
        '''
        lines = []
        f = open(file_path, 'r', encoding='utf8')
        for line in f.readlines():
            line_pos = line.replace('\n', '')
            '''
            The "underscore" character has the task to concatenate continuous tokens into a work
            The "space" character has the task to segment work (to mark the boundary of two words)
            "Two lines" of code bellow has the task to fix the little errors in the VLSP 2013 for Word Segmentation dataset
            These errors occur only "four times" in the "testing set" of the VLSP 2013 for Word Segmentation dataset
            Therefore, that errors will be not affected on all results because of it very very very very very small than total
            '''
            ######################################
            line_pos = line_pos.replace('_ ', ' ')
            line_pos = line_pos.replace(' _', ' ')
            ######################################
            lines.append(self.normalize_accent(line_pos))
        f.close()
        return lines
    
    
    def read_pos_training_corpus(self, file_path):
        '''
        Input:
        Output:
        '''
        f = open(file_path, 'r', encoding='utf8')

        list_stns, list_POSs = [], []
        words_tmp, POSs_tmp  = [], []
        lbls, id2lbl         = {}, {}

        for line in f.readlines():
            if len(line.replace('\n', '')) == 0:
                list_stns.append(words_tmp)
                list_POSs.append(POSs_tmp)
                words_tmp, POSs_tmp = [], []
                continue

            line_split = line.replace('\n', '').split('\t')
            words_tmp.append(self.normalize_accent(line_split[0]))
            lbl = self.normalize_accent(line_split[1])
            if lbl not in lbls:
                lbls[lbl] = len(lbls)
                
            POSs_tmp.append(lbls[lbl])
                
        f.close()
        
        for key in lbls.keys():
            id2lbl[lbls[key]] = key
        
        return list_stns, list_POSs, lbls, id2lbl
    
    
    def read_pos_test_corpus(self, file_path):
        '''
        Input:
        Output:
        '''
        f = open(file_path, 'r', encoding='utf8')

        list_stns, list_POSs = [], []
        words_tmp, POSs_tmp  = [], []

        for line in f.readlines():
            if len(line.replace('\n', '')) == 0:
                list_stns.append(words_tmp)
                list_POSs.append(POSs_tmp)
                words_tmp, POSs_tmp = [], []
                continue

            line_split = line.replace('\n', '').split('\t')
            words_tmp.append(self.normalize_accent(line_split[0]))
            POSs_tmp.append(self.normalize_accent(line_split[1]))

        f.close()
        return list_stns, list_POSs
    
    
    def add_string_to_dict(self, vocab, string):
        '''
        Put a string to a dictionary and update the counters of keys
        '''
        if string not in vocab: vocab[string] = 1
        else: vocab[string] += 1
            
            
    def syl_type(self, s):
        '''
        Return style of an input token by using "utils" object
        (All styles are described above)
        '''
        if s == self.bos:
            return self.bos
        if s == self.eos:
            return self.eos
        if s.islower():
            return self.lower
        if s.isupper():
            return self.allupper 
        if len(s) > 1 and s[0].isupper() and s[1:].islower():
            return self.upper
        return self.other
        
    
    def inVNDict(self, syl):
        '''
        Input: a string
        Output: True or False
        (Check whether a string in the Vietnamese dictionary)
        '''
        return syl in self.VNDict
    
    
    def inVNFamilyName(self, syl):
        '''
        Input: a string
        Output: True or False
        (Check whether a string is a Vietnamese family name)
        '''
        return syl in self.VNFamilyName
    
    
    def inVNMiddle(self, syl):
        '''
        Input: a string
        Output: True or False
        (Check whether a string is a Vietnamese middle name)
        '''
        return syl in self.VNMiddle
    
    
    def compute_ratios(self, training_sentences):
        # Counters of some n-grams patterns
        n_gram_1     = {'e1_gram-1_gram_n0_pre': {}, 'e2_gram-1_gram_n0': {}}
        pos_1_gram   = {}
        head_sfx     = {}

        # Counting for some n-grams patterns
        for line in training_sentences:
            n_grams   = line.split(' ')
            n_grams_N = len(n_grams)
            for idx, n_gram in enumerate(n_grams):
                tokens          = n_gram.lower().split('_')
                tokens_original = n_gram.split('_')
                if 4 < len(tokens) and len(tokens) < 10:
                    self.VNDict.update({n_gram.replace('_', ' ').lower()})
                if len(tokens) == 1:
                    if idx < n_grams_N - 1:
                        next_word = n_grams[idx + 1]
                        if len(next_word.split('_')) > 1:
                            if self.syl_type(tokens_original[0]) == self.lower:
                                self.add_string_to_dict(n_gram_1['e1_gram-1_gram_n0_pre'], tokens[0])
                        elif self.syl_type(next_word) in [self.lower, self.allupper, self.upper]:
                            if self.syl_type(tokens_original[0]) == self.lower:
                                self.add_string_to_dict(n_gram_1['e1_gram-1_gram_n0_pre'], tokens[0])
                if len(tokens) > 1:
                    if self.syl_type(tokens_original[0]) == self.lower:
                        self.add_string_to_dict(n_gram_1['e2_gram-1_gram_n0'], tokens[0])
                if len(tokens) == 3 or len(tokens) == 4:
                    head = ' '.join(tokens[:-1])
                    sfx = tokens[-1]
                    if self.inVNDict(head) and not self.inVNDict(' '.join(tokens)):
                        if all(self.syl_type(t_original) == self.lower for t_original in tokens_original):
                            if not self.inVNFamilyName(tokens[0]):
                                self.add_string_to_dict(pos_1_gram, sfx)
                                if sfx not in head_sfx:
                                    head_sfx[sfx] = {head}
                                else:
                                    head_sfx[sfx].update({head})
                                    
                            
        avg_pos_1_gram = 0
        for key in pos_1_gram:
            avg_pos_1_gram += pos_1_gram[key]

        pos_final = []
        if len(pos_1_gram) > 0:
            avg_pos_1_gram = avg_pos_1_gram/len(pos_1_gram)
            
            for key in pos_1_gram:
                if pos_1_gram[key] > avg_pos_1_gram and len(head_sfx[key]) > 1:
                    pos_final.append(key)
                    
        
        tmp_ratios_2 = {}
        avg_ratios_2 = 0
        ratios_2     = {}
                
        for key in n_gram_1['e1_gram-1_gram_n0_pre'].keys():
            if key in n_gram_1['e2_gram-1_gram_n0']:
                quantity            = n_gram_1['e1_gram-1_gram_n0_pre'][key]+n_gram_1['e2_gram-1_gram_n0'][key]
                tmp_ratios_2[key]   = [n_gram_1['e1_gram-1_gram_n0_pre'][key]>n_gram_1['e2_gram-1_gram_n0'][key],quantity]
                avg_ratios_2       += quantity
        
        if len(tmp_ratios_2) > 0:
            avg_ratios_2 = avg_ratios_2/len(tmp_ratios_2)

            for key in tmp_ratios_2:
                if tmp_ratios_2[key][1] > avg_ratios_2:
                    if tmp_ratios_2[key][0]:
                        ratios_2[key] = tmp_ratios_2[key][0]
                
        return {'sep': ratios_2, 'sfx': pos_final}
    
    
    def extract_training_sentence(self, sentence):
        '''
        Input: a sentence (with "underscores" character)
        Output: the list of syllabus and labels (0 for "space" and 1 for "underscore" after each token)

        For instance:
            + Input: bùng_phát việc khai_thác tự_do mỏ sắt Trại_Bò
            + Output:
                - syls: ['bùng', 'phát', 'việc', 'khai', 'thác', 'tự', 'do', 'mỏ', 'sắt', 'Trại', 'Bò']
                - lbls: [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]

        We have noted that the "label" at the end of a sentence always is "space"! (And no "label" at the begin of sentence)
        '''
        syls, lbls, cur_idx = [], [], 0
        stn = sentence + ' '
        for idx, character in enumerate(stn):
            if character == ' ' or character == '_':
                if character == ' ':
                    lbls.append(0)
                else:
                    lbls.append(1)
                syls.append(stn[cur_idx:idx])
                cur_idx = idx + 1
                                        
        return syls, lbls
    
    
    def extract_syls_windows(self, syls, lbls, window_size=8):
        '''
        Input: the list of syllabus and labels of training sentence
        Output: all syllabus windows (by padding at the begin and end of sentence and sliding!)
        '''
        bi_lbl = copy.deepcopy(lbls)
        for i in range(window_size):
            syls = [self.bos] + syls + [self.eos]
            bi_lbl = [0] + bi_lbl + [0]

        return [[syls[i-window_size:i+window_size+1], bi_lbl[i-window_size:i+window_size+1]]\
                for i in range(window_size, len(syls)-window_size)]
    
    
    def extract_syls_test_windows(self, syls, window_size=8):
        '''
        Input: the list of syllabus of testing sentence
        Output: all syllabus windows (by padding at the begin and end of sentence and sliding!)
        '''
        for i in range(window_size):
            syls = [self.bos] + syls + [self.eos]

        return [syls[i-window_size:i+window_size+1] for i in range(window_size, len(syls)-window_size)]
    
    
    def get_support(self, true_line, pred_line):
        '''
        Input:
            + The dictionary of "true" and "predicted" line
            + The format of key is: [the begin position of word]_[the end position of word]
            + The format of value is: string of word

        Output:
            + Corrected prediction satisfies: both the predicted and true pair have the same key and value
            + Number of corrected words are segmented, number of words in "predicted" line, and number of word in "true" line
        '''
        nb_correctly_segmented = 0

        for key in pred_line.keys():
            if key in true_line and pred_line[key] == true_line[key]:
                nb_correctly_segmented += 1

        return nb_correctly_segmented, len(pred_line), len(true_line)
    
    
    def get_support_details(self, true_line, pred_line, unseen_words_sfx):
        '''
        Input:
            + The dictionary of "true" and "predicted" line
            + The format of key is: [the begin position of word]_[the end position of word]
            + The format of value is: string of word

        Output:
            + Corrected prediction satisfies: both the predicted and true pair have the same key and value
            + Number of corrected words are segmented, number of words in "predicted" line, and number of word in "true" line
        '''
        nb_correctly_segmented = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        nb_pred                = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        nb_true                = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        
        for key in pred_line.keys():
            nb_syls = len(pred_line[key].split('_'))
            if 10 <= nb_syls:
                nb_syls = '10-'
            elif 5 <= nb_syls and nb_syls <= 9:  
                nb_syls = '5-9'
            elif nb_syls in [3,4]:
                if pred_line[key].lower() in unseen_words_sfx:
                    nb_syls = str(nb_syls) + 'b'
                else:
                    nb_syls = str(nb_syls) + 'a'
            else:
                nb_syls = str(nb_syls)
                
            nb_pred[nb_syls] += 1
                
        for key in true_line.keys():
            nb_syls = len(true_line[key].split('_'))
            if 10 <= nb_syls:
                nb_syls = '10-'
            elif 5 <= nb_syls and nb_syls <= 9:  
                nb_syls = '5-9'
            elif nb_syls in [3,4]:
                if true_line[key].lower() in unseen_words_sfx:
                    nb_syls = str(nb_syls) + 'b'
                else:
                    nb_syls = str(nb_syls) + 'a'
            else:
                nb_syls = str(nb_syls)
                
            nb_true[nb_syls] += 1

        for key in true_line.keys():
            nb_syls = len(true_line[key].split('_'))
            if 10 <= nb_syls:
                nb_syls = '10-'
            elif 5 <= nb_syls and nb_syls <= 9:  
                nb_syls = '5-9'
            elif nb_syls in [3,4]:
                if true_line[key].lower() in unseen_words_sfx:
                    nb_syls = str(nb_syls) + 'b'
                else:
                    nb_syls = str(nb_syls) + 'a'
            else:
                nb_syls = str(nb_syls)
            
            if key in pred_line and pred_line[key] == true_line[key]:
                nb_correctly_segmented[nb_syls] += 1

        return nb_correctly_segmented, nb_pred, nb_true
    
    
    def compute_score_details(self, list_stn, list_predict, unseen_words_sfx):
        nb_correct, nb_output, nb_ref = 0, 0, 0

        nb_correct = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        nb_output  = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        nb_ref     = {'1': 0, '2': 0, '3a': 0, '3b': 0, '4a': 0, '4b': 0, '5-9': 0, '10-': 0}
        precision  = {}
        recall     = {}
        f1_score   = {}

        for idx, stn in enumerate(list_stn):       
            pred_sentence = list_predict[idx]
            n_c, n_p, n_r = self.get_support_details(self.exact_wordboundary(stn), self.exact_wordboundary(pred_sentence),\
                                                     unseen_words_sfx)

            for key in nb_correct.keys():
                nb_correct[key] += n_c[key]
                nb_output[key]  += n_p[key]
                nb_ref[key]     += n_r[key]

        for key in nb_correct.keys():
            if nb_output[key] > 0:
                precision[key]  = 100*nb_correct[key]/nb_output[key]
            else:
                precision[key]  = 0

            if precision[key] > 0:
                recall[key]     = 100*nb_correct[key]/nb_ref[key]
            else:
                recall[key]     = 0

            if precision[key]+recall[key] > 0:
                f1_score[key]   = 2*precision[key]*recall[key]/(precision[key]+recall[key])
            else:
                f1_score[key]   = 0

        performance_detail      = {'precision': precision, 'recall': recall, 'f1_score': f1_score,\
                                   'nb_correct': nb_correct, 'nb_output': nb_output, 'nb_ref': nb_ref}

        nb_correct_total        = sum(nb_correct.values())
        nb_output_total         = sum(nb_output.values())
        nb_ref_total            = sum(nb_ref.values())
        precision_total         = 100*nb_correct_total/nb_output_total
        recall_total            = 100*nb_correct_total/nb_ref_total
        f1_score_total          = 2*precision_total*recall_total/(precision_total+recall_total)

        performance_total       = {'precision': precision_total, 'recall': recall_total, 'f1_score': f1_score_total}

        return performance_detail, performance_total
    
    
    def exact_wordboundary(self, line):
        '''
        Input:
            + A sentence contains underscore characters

        Output:
            + The dictionary of the input line
            + The format of key is: [the begin position of word]_[the end position of word]
            + The format of value is: string of word
        '''
        tokens = line.split()
        words  = {}
        idx    = 1
        for token in tokens:
            length = len(token.split('_'))
            if length > 1:
                words[str(idx) + '-' + str(idx + length - 1)] = token
            else:
                words[str(idx + length - 1)] = token
            idx = idx + length

        return words
    
    
    def fill_underscore(self, syls, lbls):
        '''
        This process is opposite with "extract_training_sentence" function
        We fill "underscore" or "space" base on their labels
        '''
        output = ''
        for idx, word in enumerate(syls):
            output = output + word
            if lbls[idx] == 0:
                output = output + ' '
            elif lbls[idx] == 1:
                output = output + '_'

        return output[:-1]
    
    
    def B_I_O_to_underscore_space(self, id2lbl, pos, stn):
        two_lbl = []
        for idx, lbl in enumerate(pos):
            if idx == len(pos) - 1:
                two_lbl.append(0)
            else:
                if id2lbl[lbl] in ['B_W', 'I_W'] and id2lbl[pos[idx + 1]] == 'I_W':
                    two_lbl.append(1)
                else:
                    two_lbl.append(0)

        return self.fill_underscore(stn, two_lbl)
    
    
    def extract_training_pairs(self, training_sentences):
        X, Y = [], []        
        for sentence in training_sentences:
            syls, lbls   = self.extract_training_sentence(sentence)
            syls_windows = self.extract_syls_windows(syls, lbls)
            X.extend(syls_windows)
            Y.extend(lbls)

        return X, Y
    
       
    def predict_list_of_sentence_ws(self, ws, NUM_PROCESSES, list_stn, get_support, has_underscore=False):
        NUM_JOBS            = len(list_stn)        
        nor_list            = [self.normalize_accent(line) for line in list_stn]
        
        predicted_sentences = [None for i in range(NUM_JOBS)]
        
        if has_underscore:
            nb_correct, nb_output, nb_ref = 0, 0, 0
            nb_stn_correct = 0
            raw_list = [line.replace('_',' ') for line in nor_list]
        else:
            raw_list = nor_list
            
        def work(begin, end, return_dict):
            for i in range(begin, end):
                sentence     = raw_list[i]
                syls         = sentence.split(' ')
                syls_windows = self.extract_syls_test_windows(syls)
                y, y_lbl     = [], [0, 0, 0, 0, 0, 0, 0, 0]

                for j in range(len(syls_windows)):
                    y_tmp = ws['model'].predict(ws['vectorizer'].transform([[syls_windows[j], y_lbl]]))
                    y_lbl.extend(y_tmp)
                    y.extend(y_tmp)
                    y_lbl = y_lbl[1:]
                    
                return_dict[i] = self.fill_underscore(syls, y)
        
        if NUM_PROCESSES > 0:
            with Manager() as manager:
                return_dict = manager.dict()
                processes  = []
                batch_size = int(NUM_JOBS/NUM_PROCESSES)
                for i in range(NUM_PROCESSES):
                    begin = i*batch_size
                    end = begin + batch_size

                    if i == NUM_PROCESSES - 1:
                        end = NUM_JOBS
                    processes.append(mp.Process(target=work, args=(begin, end, return_dict)))

                for p in processes:
                    p.daemon = True
                    p.start()

                for p in processes:
                    p.join()
                    p.terminate()

                for k, v in return_dict.items():
                    predicted_sentences[k] = v
        else:
            return_dict = {}
            work(0, NUM_JOBS, return_dict)
            for k, v in return_dict.items():
                predicted_sentences[k] = v

        for idx, p_sentence in enumerate(predicted_sentences):
            if has_underscore:
                n_c, n_p, n_r = self.get_support(self.exact_wordboundary(nor_list[idx]), self.exact_wordboundary(p_sentence))
                nb_correct   += n_c
                nb_output    += n_p
                nb_ref       += n_r
                if n_c == n_p and n_c == n_r:
                    nb_stn_correct += 1
        
        if has_underscore:
            precision = nb_correct/nb_output
            recall    = nb_correct/nb_ref
            if precision+recall > 0:
                f1_score = 2*precision*recall/(precision+recall)
            else:
                f1_score = 0
            
            if get_support:
                return predicted_sentences, [nb_output, nb_ref, nb_correct]
            else:
                return predicted_sentences, [precision, recall, f1_score]
        
        return predicted_sentences
