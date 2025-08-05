## Type Correctness for Field Annotations

Type correctness is secured thanks to venus.validators, it checks for type names and validates whether all type names are valid.
* Validation searches in typing, typing_extensions and builtins first to reach type.
* TODO: Import names from workspace (imported names in current frame)
* If it is a custom type, it will make a search in allowed namespaces.
* After validation, all the types names will convert into python types
  * Example: 'Literal[1, 2]' will evaluate to being typing.Literal[1,2] as a python type!
* Validation also determines marker schema (fields and annotations) to create pydantic models!

## User Inputs - Input Fields

User inputs can be defined with << input_field: annotation >> pattern.
* Annotation is not a must, defaulted to string or LLM determines based on context (use Any to determine field type by LLM)
  * Example: When u type "<< number_of_days_in_week: Any >>", LLM probably determines type is int (depends on LLM)
* Inputs can be retrieved from stdin, environment variables
* You can use input utilities like this:
  * validators.injectors.stdin -> creates a stdin process to retrieve value
    * Question('what is square of <<::number: int>>') # set stdin injector in syntax
    * Question('what is square of <<number: int>>', number=stdin) # set stdin injector manually
  * validators.injectors.env -> retrieves field value from env variables
    * Question('what is square of <<:number: int>>') # # set env injector in syntax
    * Question('what is square of <<number: int>>', number=env) # # set env injector manually
  * validators.meta.Injector -> retrieves field from custom injector
    * Injector is an Abstract class for defining injectors.
    * Injector infrastructure define core methods that 3 of them optional, one is required
    * ?? retrieve(self: Injector) -> Any: Returns retrieved value, implemente retriever mehcanism here
      * i.e. Reading from file, json, scraping data is responsibility of `retrieve`
      * Applies this approach in order:
        * Retrieve value from source and return.
    * ?? validate(self, value) -> tuple[list, bool]: Implement validation logic here, must return errors list and a boolean
      * Applies this approach in order:
        * Define a error list
        * Make your validations and add error messages into error list
        * return errors, validation (validation = len(errors) == 0)
    * ?? process(self:  Injector) -> FinalValueT: Returns validated and processed final value
      * This is the proccessor and core validator of Injector
      * Applies this approach in order:
        * call retrieve: Retrieve the value
        * `process` and `transform` retrieved value
          * for example, cast `stdin/env` inputs into correspond expressions
        * call `validate`: Validate retrieved value
      * Assertions and exceptions must be raised here.
      * If errors list that comes from validate function has members, it will raise that errors with an `exception`
      * If validation ends with no errors, returns processed value
    * ** inject(self) -> FinalValueT: Injects value into Injector.final_value variable
    * Applies this approach in order:
      * Set injector.final_value = injector.process()
    
    * class jsoninject(Injector): pass # pseudo injector, pretend it is valid
    * Question('what is square of <<number: int>>', number=jsoninject, path='config.json', attrpath='values.numbers[-1].number')
    * This pseudo example retrieves json data from path and selects number field by applying attrpath.

* Each input mechanism would validate type correctness for fields (TODO)
* Inputs can also be specified via Marker definition
  * Question('what is square of <<number: int>>', number=27)


## Context Aware Prompts ## 

Context Aware Prompts is a protocol that allows to talk prompts with each other thanks to Marker pattern.
Prompts know the context, any output field of a prompt, can be input field of next markers. Streamer stores the invokes in a state Markers that can reach when them invoked.

Examples:

```py

from venus.markers.streams import vStream

streamer = vStream('openai:gpt-4o')

async def main():
  async for stream in streamer.invoke(
                    'What is the <capital> of <<country>>?',
                    'What is the <population> of <<capital>> city?',
                     country='Turkiye'
                  ):
                     if hasattr(stream.output, 'capital'):
                        print('Population of ', stream.output.capital, end='')
                     else:
                        print('is ', stream.output.population)
import anyio
if __name__ == '__main__':
    anyio.run(main)
```


## Model Outputs - Output Fields

Model outputs can be defined in directly as textual in marker string with their type annotations.

* Annotation is a must for output fields to generate structured outputs, if you don't know the type,  use `contextual`?? or `Any`, then LLM will find better type for the query and will replace it with new type.
* Output fields converted into pydantic models in runtime, because we are working on textual markers :)
* Output fields can be defaulted via `<field: annotation = OutputDefault>` (TODO)


## LLM Invoking
* Single marker invoking
* Streamed bulk marker invoking
* Using clarified marker as base prompt that gives more context to LLM
