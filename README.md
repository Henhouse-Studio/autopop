# **AutoPop**

This repository is a chat-based database merging/retrieval system.

## **Setup**

To run this code, first do the following:

```

git clone gregorygo12/autopop

```

Afterwards, you need to install the dependencies:

```

# For the context-matcher
cd context-match
conda env create -f environment.yml

# For the image-matcher
cd image-match
conda env create -f environment.yml

```

## **Running**

For running the code, just do the following:

```

# For the context-matcher
streamlit run context-match/app.py

# For the image-matcher
cd image-match
python main.py

```