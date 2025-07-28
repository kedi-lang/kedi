# TODO 

* Define marker syntax
* Extract fields and types
* Validate type names
* Transform string types to correspond python types 
* Implement pydantic model builder from fields and their types, defaults if exists
* Use that models to generate structured output based on marker topics

# Scenario

```python
from venus import Venus
from venus.markers.qa import Question
from venus.helpers.io import user_input

qa = Question('What is the weather in <<city: str>> and <when: date.timedelta> weather would be better?', city=user_input)


#> Marker processed and validated!

#>> Enter the `city`:
#> 1
#> City must be a valid string but got `int`
#> Program terminate

#>> Enter the `city`:
#> Istanbul

agent = Venus('openai:gpt-4o')

result = agent.invoke(qa, template='The weather would be better after {when}')

when = repr(result.when) # 3 hours
output = f'The weather would be better after {when}'

assert output == result.output
```
