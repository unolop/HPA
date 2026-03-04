def clean_logprobs(logprobs_data):
    return {
        "content": [
            {
                "token": t['token'],
                "logprob": t['logprob'],
                "top_logprobs": [
                    {"token": tl['token'], "logprob": tl['logprob']}
                    for tl in t['top_logprobs']
                ]
            }
            for t in logprobs_data['content']
        ]
    }
