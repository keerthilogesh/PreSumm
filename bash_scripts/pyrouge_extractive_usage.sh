#!/usr/bin/env bash
# [-a (evaluate all systems)]
# [-c cf]
# [-d (print per evaluation scores)]
# [-e ROUGE_EVAL_HOME]
# [-h (usage)]
# [-H (detailed usage)]
# [-b n-bytes|-l n-words]
# [-m (use Porter stemmer)]
# [-n max-ngram]
# [-s (remove stopwords)]
# [-r number-of-samples (for resampling)]
# [-2 max-gap-length (if < 0 then no gap length limit)]
# [-3 <H|HM|HMR|HM1|HMR1|HMR2> (for scoring based on BE)]
# [-u (include unigram in skip-bigram) default no)]
# [-U (same as -u but also compute regular skip-bigram)]
# [-w weight (weighting factor for WLCS)]
# [-v (verbose)]
# [-x (do not calculate ROUGE-L)]
# [-f A|B (scoring formula)]
# [-p alpha (0 <= alpha <=1)]
# [-t 0|1|2 (count by token instead of sentence)]
# [-z <SEE|SPL|ISI|SIMPLE>]
# <ROUGE-eval-config-file> [<systemID>]

# extractive
C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -e C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m C:\Users\keert\AppData\Local\Temp\tmp_gkh_qvk\rouge_conf.xml
# abstractive
C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -e C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m C:\Users\keert\AppData\Local\Temp\tmp5q9k6vvg\rouge_conf.xml
