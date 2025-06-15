#AI Horde OpenAI API

This is a [Free and Open Source microservice](https://github.com/Haidra-Org/horde-openai-proxy) providing an REST API interface to the [AI Horde](https://aihorde.net) following the Open AI API specification. This will allow you to use the LLM croudsourced compute for your projects and entertainment using software which is not otherwise integrating with the [AI Horde API](https://aihorde.net/api). If you like this service, consider [joining the AI Horde yourself](https://github.com/Haidra-Org/AI-Horde/blob/main/README_StableHorde.md#joining-the-horde)!

You can read our [full API documentation](/docs) for this service. 

Please [register an AI Horde account to get your own API key](https://aihorde.net/register), or use the anonymous API key `0000000000` at the lowest priority.

This microservice has a lot of limitations over the direct [AI Horde API]((https://aihorde.net/api)). For one you cannot pass as many specialized parameters to your generation. You also don't have full visibility on the volunteer compute, so we have to take some assumptions to make this work without too much disruption to the end user. Significantly, your context window and your max tokens generation will be capped to the maximum available workers serving that particular model on the AI Horde.

**Important Note**: This microservice is a pilot and might be restricted or expanded in the future based on its popularity and impact on the AI Horde service resources. Please use it responsibly.

The standard AI Horde [Terms and Conditions](https://aihorde.net/terms) and [Privacy policy](https://aihorde.net/privacy) apply to this service.

