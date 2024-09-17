# Volvo POC

POC investigating diverse sources of information and models.

**Place all pdfs that need to be processed in the pdf-folders folder**

## Manual Start

### Environment
- python>=3.10


```bash
pip install -r requirements.txt
```

### LLM Models

#### Use OpenAI

Set 
```
USE_OPENAI=TRUE
USE_OPENAI_EMBED=TRUE
OPENAI_API_KEY="sk-key-string"
```
in the [.env](.env) 


Alternatively if `USE_OPENAI_EMBED=FALSE` the embedding defaults to local embedding 
with  `SentenceTransformer` and [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)

#### Use VLLM
```shell
USE_VLLM=TRUE
```
```bash
pip install vllm
```


#### Use Anthropic
If both Open AI LLM and VLLM are not used then the default is Anthropic.
Set your 
```shell
ANTHROPIC_API_KEY="sk-ant-xxx"
```
### Data Preparation

#### Parse the pdfs and add model in database

```
python sddb.py --init
```

#### Randomly generate N questions based on the content of the PDF

```
python sddb.py --questions_num 20
```

#### Prepare data and generate candidate questions

```
python sddb.py --init --questions_num 20
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


## APP
### Candidate Questions

When the `CANDIDATE_QUESTIONS` under `.env` points to an existing file, an additional tab `Candidate Questions` page for candidate questions will appear.

`CANDIDATE_QUESTIONS` defaults to `questions.txt`

So if you want to deploy a app with the `Candidate Questions` tab page.

You can save the list of questions you need under this file

The file format is as follows, one question per line
```
What components are included in the exhaust aftertreatment system?
When is the air suspension system activated?
What happens if Active Grip Control and the Traction Control System were off when the truck starts again?
How is the instrument lighting automatically adjusted according to the ambient light?
How should the new filter be screwed on?
What are the options available in the radio player?
Why is it important to clean the radiator with extreme caution?
How can I find the Distance Alert setting?
What conditions need to be fulfilled in order to start manual regeneration?
What are the four positions of the nozzles on the driver's side?
What does the driveline warranty cover?
How can the trucks parking brake and the service brake on any connected trailer be braked gradually while driving?
What are the different units of measurement for fuel consumption in the instrument display and the side display?
What is the maximum freezing-point depression for concentrated coolant?
How should the oil be filled in the gearbox?
What must be the function mode of the electrical system in order to disconnect the batteries?
How can the cargo weight be reset to zero? 
How do you generate a new report when automatic pre-trip check is disabled?
What is the purpose of turning the hydraulic valve anticlockwise to the stop position?
What functions does the control panel for the infotainment system have?
```
