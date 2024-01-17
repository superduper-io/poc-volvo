# Volvo With SuperDuperDB

## Manual start

### Environment
- python>=3.10


```bash
pip install -r requirements.txt
```

### If use OpenAI

```
pip install openai
```

Change .env

```
USE_OPENAI=TRUE
```

### If use opensource model

```bash
pip install vllm
```

## Data preparation

python build.py
```
```


## Start streamlit app
streamlit run app.py
```bash
```


## Automatically deploy to aws


Install a deploy framework

```bash
pip install ai-jobdeploy
```
