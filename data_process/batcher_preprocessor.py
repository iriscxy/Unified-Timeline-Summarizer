def plain(data, FLAGS):
    article_text = data[FLAGS.json_input_key]
    abstract_text = data[FLAGS.json_target_key]
    if isinstance(abstract_text, list):
        abstract_text = ' '.join(abstract_text)
    if isinstance(article_text, list):
        article_text = ' '.join(article_text)
    return article_text, abstract_text


def timeline(data, FLAGS):
    article_text = data[FLAGS.json_input_key]
    abstract_text = data[FLAGS.json_target_key]
    if isinstance(abstract_text, list):
        abstract_text = ' '.join(abstract_text)
    sentences = data[FLAGS.json_sent_key]
    extract_label = data['extract_label']
    # sentences = [0]
    return article_text, abstract_text, sentences, extract_label
