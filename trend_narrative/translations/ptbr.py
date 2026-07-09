"""
Brazilian Portuguese string catalog for narrative generation.

Strings use ICU MessageFormat ``select`` syntax for grammatical agreement
where singular/plural subject agreement changes the verb form.
"""

STRINGS: dict[str, object] = {
    # Brazilian Portuguese uses ',' as decimal separator and a space before '%'.
    # Suffixes intentionally use Portuguese abbreviations for large numbers.
    "number_format": {
        "decimal_sep": ",",
        "percent_template": "{value} %",
        "suffixes": ["", " mil", " mi", " bi", " tri"],
    },

    # Portuguese plurals are not reliable enough for a universal suffix.
    "time_unit_fallback_plural_suffix": "",

    # Time unit singular/plural forms, keyed by the English `time_unit` arg
    # passed to relationship narrative APIs.
    "time_units": {
        "year": ("ano", "anos"),
        "month": ("mês", "meses"),
        "quarter": ("trimestre", "trimestres"),
        "week": ("semana", "semanas"),
        "day": ("dia", "dias"),
    },

    # Grammatical gender of each time unit, used by "timing_same".
    "time_unit_genders": {
        "year": "masculine",
        "month": "masculine",
        "quarter": "masculine",
        "week": "feminine",
        "day": "masculine",
    },

    # Direction words.
    "unknown": "desconhecido",
    "remained_stable": (
        "{number, select, "
        "singular {permaneceu estável} "
        "other {permaneceram estáveis}}"
    ),
    "increased": "{number, select, singular {aumentou} other {aumentaram}}",
    "decreased": "{number, select, singular {diminuiu} other {diminuíram}}",

    # Correlation strength labels
    "strength_no": "nenhuma",
    "strength_weak": "fraca",
    "strength_moderate": "moderada",
    "strength_strong": "forte",
    "strength_very_strong": "muito forte",

    # narrative.py — volatility fallbacks
    "vol_low": (
        "{number, select, "
        "singular {{metric} permaneceu muito estável e dentro de uma faixa estreita.} "
        "other {{metric} permaneceram muito estáveis e dentro de uma faixa estreita.}}"
    ),
    "vol_moderate": (
        "{number, select, "
        "singular {{metric} apresentou flutuações moderadas em torno de uma média constante.} "
        "other {{metric} apresentaram flutuações moderadas em torno de uma média constante.}}"
    ),
    "vol_high": (
        "{number, select, "
        "singular {{metric} apresentou volatilidade sem uma direção clara.} "
        "other {{metric} apresentaram volatilidade sem uma direção clara.}}"
    ),

    # narrative.py — single segment
    # {pct_change} is pre-formatted (sign, decimals, %) by _format_percent.
    "single_segment": (
        "entre {start_year} e {end_year}, "
        "{metric} {direction} em {change} "
        "({pct_change}), mantendo uma trajetória consistente."
    ),

    # narrative.py — multi-segment
    "trend_upward": "uma tendência de alta",
    "trend_downward": "uma tendência de queda",
    "path_upward": "de alta",
    "path_downward": "de queda",
    "first_segment": (
        "{number, select, "
        "singular {De {start_year} a {end_year}, {metric} apresentou {trend_phrase}.} "
        "other {De {start_year} a {end_year}, {metric} apresentaram {trend_phrase}.}}"
    ),
    "transition_prefixes": [
        "A tendência então mudou,",
        "Essa trajetória mudou novamente,",
        "Depois,",
    ],
    "peak_reversal": (
        "atingindo um pico em {year} "
        "antes de se inverter em queda."
    ),
    "low_recovery": (
        "atingindo um mínimo em {year} "
        "seguido por uma recuperação."
    ),
    "continuing": "continuando sua trajetória {path_phrase} até {year}.",

    # relationship_narrative.py — comovement
    "period_from_to": "de {start} a {end}",
    "unable_to_analyze": (
        "Não foi possível analisar a relação entre {x} e {y}."
    ),
    "no_data_available": (
        "A relação entre {x} e {y} "
        "não pode ser determinada porque os dados {y_gen} não estão disponíveis."
    ),
    "single_observation": (
        "{period}, {ref_name} {ref_dir} "
        "({ref_start} a {ref_end}), "
        "com apenas uma observação {comp_name_gen} ({comp_start})"
    ),
    "stable_comparison": (
        "{number, select, "
        "singular {{period}, {ref_name} {ref_dir} ({ref_start} a {ref_end}) enquanto {comp_name} permaneceu estável ({comp_start})} "
        "other {{period}, {ref_name} {ref_dir} ({ref_start} a {ref_end}) enquanto {comp_name} permaneceram estáveis ({comp_start})}}"
    ),
    "both_same_direction": "ambos se movendo na mesma direção",
    "opposite_directions": "movendo-se em direções contrárias",
    "comovement_with_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} a {ref_end}) "
        "enquanto {comp_name} {comp_dir} "
        "({comp_start} a {comp_end}), {relationship}"
    ),
    "comovement_no_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} a {ref_end}) "
        "enquanto {comp_name} {comp_dir} "
        "({comp_start} a {comp_end})"
    ),
    "limited_data_caveat": (
        "Com dados limitados sobre {comp_name}, "
        "não foi possível estabelecer uma relação estatística."
    ),

    # relationship_narrative.py — lagged correlation
    "timing_same": (
        "{gender, select, "
        "feminine {na mesma {time_unit}} "
        "other {no mesmo {time_unit}}}"
    ),
    "timing_lagged": "cerca de {lag} {time_unit_form} depois",
    "no_reliable_relationship": (
        "Nenhuma relação confiável foi detectada entre as variações {x_gen} "
        "e {y_gen}. "
    ),
    "weak_pattern": (
        "Embora os dados sugiram uma tendência {sign} {strength} "
        "(r={corr}), isso pode ter ocorrido por acaso "
        "devido ao tamanho limitado da amostra (n={n_pairs} pares de variação, p={p_val})."
    ),
    "no_association": (
        "As variações de uma série não parecem estar associadas às variações da outra, "
        "com base em {n_pairs} comparações {time_unit_comparison}."
    ),
    "no_association_with_lag": (
        "As variações de uma série não parecem estar associadas às variações da outra "
        "em nenhuma defasagem testada (0-{max_lag} {time_unit_form}), "
        "com base em {n_pairs} comparações {time_unit_comparison}."
    ),
    "significant_finding": (
        "Quando {leader} "
        "{leader_number, select, singular {aumenta} other {aumentam}}, "
        "{follower} "
        "{follower_number, select, singular {tende} other {tendem}} a "
        "{direction_word} {timing}. "
        "Esta é uma relação {strength} (r={corr}) "
        "e estatisticamente confiável (p={p_val}), "
        "com base em {n_pairs} comparações {time_unit_comparison}."
    ),

    # relationship_narrative.py — insufficient data
    "insufficient_data": (
        "A relação entre {x} e {y} "
        "não pode ser determinada devido à disponibilidade limitada de dados."
    ),

    # Positive / negative labels for correlation sign
    "positive": "positiva",
    "negative": "negativa",
    "increase": "aumentar",
    "decrease": "diminuir",
}
