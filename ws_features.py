import sklearn.feature_extraction.text as text # for building features extraction


class WSCountVectorizer(text.CountVectorizer):
    def __init__(self, utils, ratios, feature_set):
        '''
        Inherit the "CountVectorizer" class from the "sklearn" library
        '''
        super(WSCountVectorizer, self).__init__()
        self.utils        = utils
        self.ratios       = ratios
        self.feature_set  = feature_set
        
        
    def build_analyzer(self):
        '''
        Override the "build_analyzer" method of original "CountVectorizer"
        '''
        return lambda doc: self.features_extraction(doc)
       
    
    def features_extraction(self, x, i=8):
        '''
        Override the "features_extraction" method of original "CountVectorizer"
        '''
        features, syls, bi_lbls = [], x[0], x[1]
        utils, off_set          = self.utils, None
        syl_type                = utils.syl_type
        syls_raw                = [syl for syl in syls]
        syls                    = [syl.lower() for syl in syls]
        
        def join(start, end):
            return ' '.join(syls[start:end])
        
        if 'base' in self.feature_set:               
            # The 1-gram feature            
            for j in range(i-2, i+2+1):
                features.append(str(j-i)+'①'+' '+syls[j])
                
            # The 2-gram feature
            for j in range(i-2, i+2-0):
                features.append(str(j-i)+'②'+join(j,j+2))
                if utils.inVNDict(join(j,j+2)):
                    features.append(str(j-i)+'③')
                elif syl_type(syls_raw[j]) != utils.lower:
                    features.append(str(j-i)+'④'+' '.join([syl_type(s_raw) for s_raw in syls_raw[j:j+2]]))
                    
            # The 3-gram feature
            for j in range(i-2, i+2-1):
                if utils.inVNDict(join(j,j+3)):
                    features.append(str(j-i)+'⑤')
                elif syl_type(syls_raw[j]) != utils.lower:
                    features.append(str(j-i)+'⑥'+' '.join([syl_type(s_raw) for s_raw in syls_raw[j:j+3]]))

            # The 4-gram feature
            for j in range(i-3, i+1):
                if utils.inVNDict(join(j,j+4)): features.append(str(j-i)+'⑦')
                    
            # features of bi-grams style
            cur, nxt      = syls_raw[i], syls_raw[i+1]
            cur_t, nxt_t  = syl_type(cur), syl_type(nxt)
            cur, nxt      = cur.lower(), nxt.lower()
            
            if cur_t == utils.upper and nxt_t == utils.upper:
                if utils.inVNFamilyName(cur):
                    features.append('⑧')
                if utils.inVNMiddle(cur):
                    features.append('⑨')
                    
            if bi_lbls[i-1] == 0 and cur_t == utils.lower and nxt_t == utils.lower and cur == nxt:
                features.append('⑩')
        
        if 'long' in self.feature_set:
            # The 5-gram feature
            for j in range(i-4, i+1):
                if utils.inVNDict(join(j,j+5)): features.append(str(j-i)+'⑪')
            # The 6-gram feature
            for j in range(i-5, i+1):
                if utils.inVNDict(join(j,j+6)): features.append(str(j-i)+'⑫')
            # The 7-gram feature
            for j in range(i-6, i+1):
                if utils.inVNDict(join(j,j+7)): features.append(str(j-i)+'⑬')
            # The 8-gram feature
            for j in range(i-7, i+1):
                if utils.inVNDict(join(j,j+8)): features.append(str(j-i)+'⑭')
            # The 9-gram feature
            for j in range(i-8, i+1):
                if utils.inVNDict(join(j,j+9)): features.append(str(j-i)+'⑮')
        
        sfx_3, sfx_4 = '', ''
        if 'sfx' in self.feature_set:
            if bi_lbls[i-1] == 1 and bi_lbls[i-2] == 0: # word contains 3 syllables
                off_set, sfx_3 = 0, 'sfx_3'
            elif bi_lbls[i-1] == 1 and bi_lbls[i-2] == 1 and bi_lbls[i-3] == 0: # word contains 4 syllables
                off_set, sfx_4 = 1, 'sfx_4'
            if off_set is not None and syls[i+1] in self.ratios['sfx']:
                features.extend(['⑯ⓐ' + (join(i-1-off_set,i+1)), '⑯ⓑ' + (syls[i+1]),
                                 '⑯ⓒ' + (syls[i+2]), '⑯ⓓ' + (syls[i+3]),
                                 '⑯ⓔ' + (syls[i-2-off_set]), '⑯ⓕ' + (syls[i-3-off_set])])
            else:
                sfx_3, sfx_4 = '', ''
                                        
        if 'sep' in self.feature_set:
            def ambiguity_template(i):
                return [
                    'ⓐ' + ''.join([str(int(utils.inVNDict(join(j,j+2)))) for j in range(i, i+4)]),
                    'ⓑ' + ''.join([str(int(utils.inVNDict(join(j,j+3)))) for j in range(i, i+3)]),
                    'ⓒ' + ''.join([str(int(utils.inVNDict(join(j,j+4)))) for j in range(i, i+2)]),
                    'ⓓ' + ''.join([str(int(utils.inVNDict(join(j,j+5)))) for j in range(i, i+1)])
                ]

            if bi_lbls[i-1] == 0 and syls[i] in self.ratios['sep'].keys():
                features.extend([('⑰' + t) for t in ambiguity_template(i+0)])
            if bi_lbls[i-1] == 1 and bi_lbls[i-2] == 0:
                features.extend([('⑱' + sfx_3 + t) for t in ambiguity_template(i-1)])
            if bi_lbls[i-1] == 1 and bi_lbls[i-2] == 1 and bi_lbls[i-3] == 0:
                features.extend([('⑲' + sfx_4 + t) for t in ambiguity_template(i-2)])
            if bi_lbls[i-1] == 1 and bi_lbls[i-2] == 1 and bi_lbls[i-3] == 1 and bi_lbls[i-4] == 0:
                features.extend([('⑳' + t) for t in ambiguity_template(i-3)])
         
        
        # Return all features given syllables windows      
        return features