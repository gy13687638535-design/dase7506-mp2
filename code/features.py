"""Label-free candidate features. No references, IDs, or split names are used."""
import ast
from fractions import Fraction
import math
import re

import numpy as np
from common import parse_answer

NUMBER = re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORDS = re.compile(r'[a-z]+')
CUES = ('wait', 'however', 'assum', 'approximately', 'alternatively', 'mistake',
        'not enough', 'cannot', 'therefore', 'check', 'remainder', 'round',
        'integer', 'total', 'remaining', 'each', 'per', 'ratio', 'percent')


def numbers(text):
    values = []
    for token in NUMBER.findall(text):
        try:
            value = float(Fraction(token.replace(',', '')))
            if math.isfinite(value):
                values.append(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return values


def safe_arithmetic(expression):
    """Evaluate bounded numeric arithmetic only, never arbitrary Python code."""
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError('too complex')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
        raise ValueError('unsupported arithmetic')
    result = visit(tree)
    if not math.isfinite(result) or abs(result) > 1e15:
        raise ValueError('unbounded result')
    return result


def equation_stats(text):
    clean = text.replace(',', '').replace('\\times', '*').replace('×', '*')
    clean = clean.replace('\\cdot', '*').replace('\\div', '/').replace('÷', '/')
    for _ in range(3):
        clean = re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', clean)
    clean = clean.replace('$', '').replace('\\left', '').replace('\\right', '')
    correct, wrong = 0, 0
    pattern = r'([\d. ()+*/-]+)\s*=\s*([-+]?\d+(?:\.\d+)?)'
    for match in re.finditer(pattern, clean):
        if not any(op in match.group(1) for op in '+*/-'):
            continue
        try:
            actual = safe_arithmetic(match.group(1))
            stated = float(match.group(2))
            ok = math.isclose(actual, stated, rel_tol=1e-5, abs_tol=1e-5)
            correct += int(ok)
            wrong += int(not ok)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            pass
    return [correct, wrong, wrong/max(1, correct+wrong)]


def extract_features(bank):
    """Return named answer, style, arithmetic, and comparison feature blocks."""
    blocks = {k: [] for k in ('answer', 'style', 'arithmetic', 'comparison')}
    validity = []
    for row in bank:
        q = row['question']
        qnums = numbers(q)
        qwords = set(WORDS.findall(q.lower()))
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        vals = [float(Fraction(a)) if a is not None else 0. for a in answers]
        counts = [answers.count(a) if a is not None else 0 for a in answers]
        lengths = np.array([len(t) for t in texts], dtype=float)
        valid = [a is not None for a in answers]
        validity.append(valid)
        validvals = [v for v, ok in zip(vals, valid) if ok]
        for j, text in enumerate(texts):
            value = vals[j]
            lower = text.lower()
            words = WORDS.findall(lower)
            tnums = numbers(text)
            absval = abs(value)
            qscale = max([abs(x) for x in qnums] + [1.])
            blocks['answer'].append([
                int(valid[j]), int(value == 0), int(value < 0),
                int(value == int(value)), math.log1p(absval),
                math.log1p(absval/qscale), len(qnums),
                int(any(math.isclose(value, n, rel_tol=1e-8, abs_tol=1e-8) for n in qnums)),
                math.log1p(len(q)), math.log1p(qscale),
                int('how many' in q.lower()), int('how much' in q.lower()),
                int('%' in q or 'percent' in q.lower()),
            ])
            blocks['style'].append([
                math.log1p(len(text)), math.log1p(len(words)),
                math.log1p(text.count('\n')), text.count('**'), text.count('###'),
                text.count('\\boxed'), text.count('####'), text.count('$'),
                text.count('='), text.count('\\frac'), text.count('?'),
                len(set(words))/max(1, len(words)),
                len(set(words)&qwords)/max(1, len(qwords)),
                math.log1p(len(tnums)),
            ] + [lower.count(cue) for cue in CUES])
            blocks['arithmetic'].append(equation_stats(text))
            others = [vals[k] for k in range(4) if k != j and valid[k]]
            same_lengths = [lengths[k] for k in range(4) if answers[k] == answers[j]]
            blocks['comparison'].append([
                counts[j], max(counts), len(set(a for a in answers if a is not None)),
                sum(valid), counts[j] == max(counts),
                (sum(v < value for v in validvals)+.5*sum(v == value for v in validvals))/max(1,len(validvals)),
                math.log1p(min([abs(value-v) for v in others] or [0.])),
                lengths[j]/max(1., lengths.mean()),
                (lengths[j]-lengths.mean())/max(1., lengths.std()),
                lengths[j]/max(1., np.mean(same_lengths)),
            ])
    return {k: np.asarray(v, dtype=np.float32).reshape(len(bank), 4, -1) for k, v in blocks.items()}, np.asarray(validity)


def embedding_features(arrays):
    e = np.asarray(arrays['embeddings'], dtype=np.float32)
    e = e/np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    return e, e-e.mean(axis=1, keepdims=True)


def majority(bank):
    selected = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        counts = [answers.count(a) if a is not None else 0 for a in answers]
        selected.append(int(np.argmax(counts)))
    return selected
