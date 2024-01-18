# Volvo With SuperDuperDB

Place all pdfs that need to be processed in the pdf-folders folder

## Manual Start

### Environment
- python>=3.10


```bash
pip install -r requirements.txt
```

### LLM Model

#### Use OpenAI

```
pip install openai
```

Change .env

```
USE_OPENAI=TRUE
```

#### Use OpenSource model

```bash
pip install vllm
```

### Data preparation

```
python sddb.py --init
```


### Start Streamlit app
```bash
streamlit run app.py
```


## Automatically Deploy To AWS


Install a deploy framework

```bash
pip install ai-jobdeploy awscli
```

```
jd build up --template=aws --params instance_type='g4dn.xlarge',name=volvo-demo-vector-search
```
