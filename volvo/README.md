# Volvo With SuperDuperDB

Place all pdfs that need to be processed in the pdf-folders folder

## Manual start

### Environment
- python>=3.10


```bash
pip install -r requirements.txt
```

### Use OpenAI

```
pip install openai
```

Change .env

```
USE_OPENAI=TRUE
```

### Use OpenSource model

```bash
pip install vllm
```

## Data preparation

```
python build.py
```


## Start streamlit app
```bash
streamlit run app.py
```


## Automatically deploy to aws


Install a deploy framework

```bash
pip install ai-jobdeploy awscli
```

```
jd build up --template=aws --params instance_type='g4dn.xlarge',name=volvo-demo-vector-search-10008
```
