# Vietnamese Word Segmentation with SVM: Ambiguity Reduction and Suffix Capture

The implementation of UITws, as described in [our paper](https://link.springer.com/chapter/10.1007/978-981-15-6168-9_33):

    @InProceedings{10.1007/978-981-15-6168-9_33,
      author    = "Nguyen, Duc-Vu and Van Thin, Dang and Van Nguyen, Kiet and Nguyen, Ngan Luu-Thuy",
      editor    = "Nguyen, Le-Minh and Phan, Xuan-Hieu and Hasida, K{\^o}iti and Tojo, Satoshi",
      title     = "Vietnamese Word Segmentation with SVM: Ambiguity Reduction and Suffix Capture",
      booktitle = "Computational Linguistics",
      year      = "2020",
      publisher = "Springer Singapore",
      address   = "Singapore",
      pages     = "400--413",
      abstract  = "In this paper, we approach Vietnamese word segmentation as a binary classification by using the Support Vector Machine classifier. We inherit features from prior works such as n-gram of syllables, n-gram of syllable types, and checking conjunction of adjacent syllables in the dictionary. We propose two novel ways to feature extraction, one to reduce the overlap ambiguity and the other to increase the ability to predict unknown words containing suffixes. Different from UETsegmenter and RDRsegmenter, two state-of-the-art Vietnamese word segmentation methods, we do not employ the longest matching algorithm as an initial processing step or any post-processing technique. According to experimental results on benchmark Vietnamese datasets, our proposed method obtained a better {\$}{\$}{\backslash}text {\{}F{\}}{\_}{\{}1{\}}{\backslash}text {\{}-score{\}}{\$}{\$}F1-scorethan the prior state-of-the-art methods UETsegmenter, and RDRsegmenter.",
      isbn      = "978-981-15-6168-9"
    }

In case you use UITws to produce published results or incorporated it into other software, please cite our paper as described above. You can you the notebook file in this repository to train and test your Vietnamese word segmentation corpora. Additionally, you can predict from raw texts as you can see in the `Inference from raw texts.ipynb` file. We also provide two `zip` files of two best models in the `checkpoints` directory.
