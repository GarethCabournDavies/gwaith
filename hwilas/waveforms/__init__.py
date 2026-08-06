from hwilas.waveforms.bbhx_phenom import get_fd_waveform_from_bbhx


def get_fd_waveform(
    approximant,
    **waveform_params
):
    if approximant == "BBHX_PhenomHM" or approximant == "BBHX_PhenomD":
        return get_fd_waveform_from_bbhx(
            approximant=approximant,
            **waveform_params
        )
    else:
        raise NotImplementedError(
            f"Waveform approximant {approximant} not yet implemented"
        )