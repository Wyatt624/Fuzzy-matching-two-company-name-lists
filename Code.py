import re 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct


# In our function, company1 and company2 should be two pandas DataFrame with two columns(index?,comn?)

# Important variables:
# Don't set the option to keep the top n
# Set the score cut which is a lower bound of sparse matrix
# col is the column of company1 for matching work

class Fuzzy_match_similarity():
    # Some necessary inputs
    def __init__(self,company1,company2,score_cut=0.8,col='con1',year=1991):
        self.company1=company1
        self.company2=company2
        self.col=col
        self.year=year

        self.score_cut=score_cut
        
    # Function for generating all n-grams in a company name
    def ngrams(string, n=3):
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    # Function for calculating dot of two sparse matrixs and select the top n values.
    def awesome_cossim_top(A, B, ntop, lower_bound=0):
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape
    
        idx_dtype = np.int32
    
        nnz_max = M*ntop
    
        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        return csr_matrix((data,indices,indptr),shape=(M,N))
    
    # After calculating the similarity score, we select the result company name from two data lists.
    def get_matches_df(self,sparse_matrix, name_vector1,name_vector2,index_vector1,index_vector2):
        non_zeros = sparse_matrix.nonzero()
        
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]
        
        # The number of nonzero values which means how much company1 matches company2 within our conditions.
        nr_matches = sparsecols.size
        # print(nr_matches)
        index1 = np.empty([nr_matches], dtype=object)
        index2 = np.empty([nr_matches], dtype=object)
        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            index1[index] = index_vector1[sparserows[index]]
            left_side[index] = name_vector1[sparserows[index]]
            index2[index] = index_vector2[sparsecols[index]]
            right_side[index] = name_vector2[sparsecols[index]]
            similairity[index] = sparse_matrix.data[index]
        
        return pd.DataFrame({'index1':index1,
                            self.col: left_side,
                            'index2':index2,
                            'conm2': right_side,
                            'similarity_SM': similairity})
    

    # Main 
    def Main_matchfunc(self,):
        # Generate n-gram and transform to sparse matrix
        vectorizer1 = TfidfVectorizer(min_df=1, analyzer=Fuzzy_match_similarity.ngrams)
        company_names1 = self.company1[self.col].tolist()
        index_1 = self.company1['index1'].tolist()
        tf_idf_matrix1 = vectorizer1.fit_transform(company_names1)
        
        vectorizer2 = TfidfVectorizer(min_df=1, analyzer=Fuzzy_match_similarity.ngrams)
        company_names2 = self.company2.conm2.tolist()
        index_2 = self.company2['index2'].tolist()
        tf_idf_matrix2 = vectorizer2.fit_transform(company_names2)
        

        # Select the same string gram from two sparse matrix
        # Intersection elements
        con = set(vectorizer1.get_feature_names_out()).intersection(vectorizer2.get_feature_names_out())
        
        # Get the index
        v1 = pd.DataFrame(vectorizer1.get_feature_names_out(),columns=['v1'])
        con_v1 = v1[v1['v1'].isin(con)].index.tolist()

        v2 = pd.DataFrame(vectorizer2.get_feature_names_out(),columns=['v2'])
        con_v2 = v2[v2['v2'].isin(con)].index.tolist()

        # Keep the elements
        tf_idf_matrix1 = tf_idf_matrix1[:,con_v1]
        tf_idf_matrix2 = tf_idf_matrix2[:,con_v2]

        # Calculate the similarity between two sparse matrixs and keep the top n elements
        matches = Fuzzy_match_similarity.awesome_cossim_top(tf_idf_matrix1, tf_idf_matrix2.transpose(), tf_idf_matrix2.shape[0], self.score_cut)

        # Return the final result
        matches_df = Fuzzy_match_similarity.get_matches_df(self,matches,company_names1,company_names2,index_1,index_2)
        matches_df=matches_df.assign(match_way='Sparse Matrix',YEAR=self.year)
        
        matches_df['index2']=matches_df['index2'].astype('Int64')

        # Combine the match result in one list.
        mg=matches_df.groupby(['index1',self.col,'YEAR',])

        mg1=pd.concat([mg.apply(lambda x: x.conm2.tolist() if len(x.conm2.tolist())>1 else x.conm2.tolist()[0]),mg.apply(lambda x: x.index2.tolist() if len(x.index2.tolist())>1 else x.index2.tolist()[0]),mg.apply(lambda x: x.similarity_SM.tolist() if len(x.similarity_SM.tolist())>1 else x.similarity_SM.tolist()[0])],axis=1)\
            .rename(columns={0:'conm2_'+self.col,1:'index2_'+self.col,2:'similarity_'+self.col}).reset_index()

        return mg1


# Group the financial company data by year which is accessible for exacting year data.
c2_gy=c2.groupby(['YEAR'])

# Step 2: Sparse Matrix matching

# Loop year
for y in range(1991,2022,1):
    # Two company lists.
    cm1=c1_remain1.loc[c1_remain1['YEAR']==y]
    cm2=c2_gy.get_group(y)[['index2','conm2']]

    # Invokes class
    fms1=Fuzzy_match_similarity(cm1[['index1','YEAR','con1']].dropna(),cm2,0.8,col='con1',year=y)
    # Result
    res1=fms1.Main_matchfunc()

    # Invokes class
    # print(cm1[['index1','YEAR','con2']].dropna().shape[0])
    fms2=Fuzzy_match_similarity(cm1[['index1','YEAR','con2']].dropna(),cm2,0.8,col='con2',year=y)
    
    # Result
    res2=fms2.Main_matchfunc()
    # print(res2.shape[0])

    # Invokes class
    fms3=Fuzzy_match_similarity(cm1[['index1','YEAR','con3']].dropna(),cm2,0.8,col='con3',year=y)
    # Result
    res3=fms3.Main_matchfunc()

    # Concat with first step result.
    if y==1991:
        mat_res1=res1.copy()
        mat_res2=res2.copy()
        mat_res3=res3.copy()
    else:
        mat_res1=pd.concat([mat_res1,res1])
        mat_res2=pd.concat([mat_res2,res2])
        mat_res3=pd.concat([mat_res3,res3])

    print(y)
