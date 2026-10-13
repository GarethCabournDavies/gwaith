# Search workflows

The dataset examples in `examples/search/mojito_basic_search.sh` and
`examples/search/sangria_basic_search.sh` define filtering, candidate selection,
waveform fitting, subtraction and residual checks. Data, PSDs and template banks
must use matching sampling, units and waveform conventions. Configure each
example with its environment variables before running it with Bash.

- [Mojito inputs and execution](mojito_search.md)
- [Joint A/E matched filtering](mojito_joint_search.md)
- [Search rounds and physical-distance fitting](mojito_two_round_search.md)
- [Candidate statistics and residual checks](residual_statistics.md)
- [PE seed export](mbhbs_pe_export.md)

Stochastic bank generation uses `hwilas_make_stochastic_bank`. Independent
validation uses `hwilas_verify_bank validate --config CONFIG --bank-file BANK
--output-file REPORT --n 400 --nproc 4`. The `prepare`, `run` and `merge`
subcommands support cached, sharded verification. Use each command's `--help`
output for its available arguments.

The numerical filtering, inpainting and configuration utilities are available as
library modules. Successful orchestration or a detected trigger does not establish
waveform accuracy, complete search sensitivity or clean signal subtraction.
