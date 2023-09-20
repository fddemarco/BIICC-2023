def arxiv_waller_scores():
    return {
        "democrats": -0.345948606707049,
        "EnoughLibertarianSpam": -0.322594981636269,
        "hillaryclinton": -0.3027931218773805,
        "progressive": -0.2994712557588187,
        "BlueMidterm2018": -0.2977831668625458,
        "EnoughHillHate": -0.2933539740564371,
        "Enough_Sanders_Spam": -0.2929483022563205,
        "badwomensanatomy": -0.2926874460908718,
        "racism": -0.2921137058022828,
        "GunsAreCool": -0.290219904193626,
        "Christians": 0.2607635855569176,
        "The_Farage": 0.2658256024989052,
        "new_right": 0.2697649330292293,
        "conservatives": 0.2743712713632447,
        "metacanada": 0.2865165930755363,
        "Mr_Trump": 0.2895610652703748,
        "NoFapChristians": 0.2934370114397415,
        "TrueChristian": 0.3142461533194396,
        "The_Donald": 0.3351316374970578,
        "Conservative": 0.444171415963574,
    }


def arxiv_waller_ranking():
    return list(arxiv_waller_scores().keys())
