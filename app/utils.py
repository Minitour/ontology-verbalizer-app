import json

from openai import OpenAI
from verbalizer.nlp import ParaphraseLanguageModel
from verbalizer.vocabulary import Vocabulary
from verbalizer.sampler import Sampler
import pandas as pd


class LLM(ParaphraseLanguageModel):
    models = {
        "gpt-4o": {
            "input": 0.005,
            "output": 0.015
        },
        "gpt-4-0125-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-vision-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-0613": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-32k": {
            "input": 0.06,
            "output": 0.12
        },
        "gpt-3.5-turbo-0125": {
            "input": 0.0005,
            "output": 0.0015
        },
        "gpt-3.5-turbo-instruct": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "input": 0.0030,
            "output": 0.0040
        },
        "gpt-3.5-turbo-1106": {
            "input": 0.0010,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "input": 0.0015,
            "output": 0.0020
        }
    }
    default_system_message = "You are an extremely specific data expert capable of converting pseudo English sentences into a meaningful and casual paragraph without losing information. Avoid repeating information. Spell out everything don't be lazy!"
    default_prompt_template = [
        {
            "User": "x relation 1 y\nz relation 2 x\nm relation 3 n",
            "Assistant": "X has this relation 1 with Y, Z shares a relation 2 with X. Moreover, m has relation 3 with n."
        },
        {
            "User": "X is same as something that intersection of something that something that has at least 3 N and M.",
            "Assistant": "X is the same as M which has at least three N"
        },
        {
            "User": "X is a type of at least has Y relation some a M.\nX is a type of at least has Y relation some a N.\nX is a type of only has Y relation any of (a M and a N)",
            "Assistant": "X is a type of Y, M, and N"
        }
    ]

    def __init__(self, model_name, temperature, message_templates, api_key):
        self.model_name = model_name
        self.temperature = temperature
        self.message_templates = message_templates
        self.api_key = api_key
        self._in_token_usage = 0
        self._out_token_usage = 0

    def pseudo_to_text(self, pseudo_text: str, extra: str = None) -> str:
        messages = self.message_templates.copy()
        messages.append({"role": "user", "content": pseudo_text})

        response = OpenAI(api_key=self.api_key).chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        self._in_token_usage += response.usage.prompt_tokens
        self._out_token_usage += response.usage.completion_tokens
        return response.choices[0].message.content

    @property
    def cost(self) -> float:
        model_pricing = self.models.get(self.model_name) or {'input': 0.0, 'output': 0.0}

        in_tokens = self._in_token_usage / 1000
        out_tokens = self._out_token_usage / 1000

        return in_tokens * model_pricing['input'] + out_tokens * model_pricing["output"]

    @property
    def name(self) -> str:
        return self.model_name


def vocabulary_from(vocabulary: Vocabulary, relationships_to_remove: pd.DataFrame, rephrased_identifiers: pd.DataFrame):
    """
    Create vocabulary from the provided configurations.
    """
    return Vocabulary(
        graph=vocabulary.graph,
        ignore=set(relationships_to_remove["IRI"].tolist()),
        rephrased=rephrased_identifiers.set_index('IRI')['Display'].to_dict(),
    )


def replace_label(element: dict, vocabulary: Vocabulary):
    iri = element.get("root")
    element['root'] = vocabulary.get_class_label(iri)
    return element


def get_number_of_concepts(vocabulary: Vocabulary, sampler: Sampler) -> int:
    if sampler is None:
        return len(vocabulary.object_labels)

    if sampler.sample_n:
        return min(sampler.sample_n, len(vocabulary.object_labels))

    if sampler.sample_percentage:
        return int(len(vocabulary.object_labels) * sampler.sample_percentage)


def generate_code(
        file_name: str,
        relationships_to_remove: pd.DataFrame,
        rephrased_identifiers: pd.DataFrame,
        sampler: Sampler,
        llm: LLM
) -> str:
    """
    This function generates Python code to perform verbalization based on the provided configurations.
    """
    imports = ''
    code = ''

    vocabulary_parameters = ['ontology']
    verbalizer_parameters = ['vocabulary', 'patterns=patterns']
    processor_parameters = ['verbalizer', 'namespace="my_ontology"', "output_dir='./output'"]

    if not relationships_to_remove.empty:
        vocabulary_parameters.append('ignore=ignore')
        code += "ignore = {\n" + ''.join(
            map(
                lambda x: f'    "{x}",\n',
                set(relationships_to_remove["IRI"].tolist())
            ),
        ) + "}\n"

    if not rephrased_identifiers.empty:
        vocabulary_parameters.append('rephrased=rephrased')
        code += f"rephrased = {json.dumps(rephrased_identifiers.set_index('IRI')['Display'].to_dict(), indent=4, sort_keys=True)}\n"

    if not (rephrased_identifiers.empty or relationships_to_remove.empty):
        code += '\n\n'

    code += 'patterns = [owl_disjoint.OwlDisjointWith, owl_restriction.OwlRestrictionPattern, owl_first_rest.OwlFirstRestPattern]\n'

    if llm:
        verbalizer_parameters.append('language_model=model')
        imports += 'from verbalizer.nlp import ChatGptModelParaphrase\n'
        code += f"model = ChatGptModelParaphrase(api_key='your-key-here', model='{llm.name}', temperature={llm.temperature})\n"

    if sampler:
        processor_parameters.append('sampler=sampler')
        imports += 'from verbalizer.sampler import Sampler\n'
        code += f"sampler = Sampler(sample_n={sampler.sample_n}, seed=42)\n"

    imports += 'from verbalizer.verbalizer import Verbalizer\n'
    imports += 'from verbalizer.process import Processor\n'
    imports += 'from verbalizer.vocabulary import Vocabulary\n'
    imports += 'from verbalizer.patterns import owl_disjoint, owl_restriction, owl_first_rest\n'


    code += f"ontology = Processor.from_file('{file_name}')\n"
    code += f"vocabulary = Vocabulary({', '.join(vocabulary_parameters)})\n"
    code += '\n'
    code += f"verbalizer = Verbalizer({', '.join(verbalizer_parameters)})\n"
    code += f"Processor.verbalize_with({', '.join(processor_parameters)})\n"

    return imports + '\n\n' + code
