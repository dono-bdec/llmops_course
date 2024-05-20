#Attaching function call information
functions = [
    {
        "name": "city",
        "description": "Describes a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city is"},
                "description": {
                    "type": "string",
                    "description": "The city is famous for",
                },
            },
            "required": ["setup", "description"],
        },
    }
]
chain = llm | prompt | llm.bind(function_call={"name": "city"}, functions=functions)
chain.invoke('Boston')

