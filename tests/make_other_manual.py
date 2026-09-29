"""Builds a synthetic owner's manual for a FICTIONAL 'Acme Roadster 150' with a deliberately different
layout from the Royal Enfield manual, to check the agent isn't tied to one brand's formatting:
- Title Case headings in a larger bold font (no ALL CAPS)
- printed page numbers at the BOTTOM, a running footer with the model name
- 'Trouble Shooting' (two words) as a problem / cause / remedy list
- a ruled service-schedule table with 750 / 3,000 / 6,000 ... km columns and a printed legend
All values are invented for testing.
"""
import sys

import pymupdf

FOOTER = "Acme Roadster 150 - Owner's Manual"


def page(doc, number, blocks):
    p = doc.new_page(width=595, height=842)
    y = 60
    for kind, text in blocks:
        if kind == "h":
            p.insert_text((50, y), text, fontsize=15, fontname="hebo"); y += 28
        else:
            for line in text.split("\n"):
                p.insert_text((50, y), line, fontsize=10, fontname="helv"); y += 15
            y += 6
    p.insert_text((50, 815), FOOTER, fontsize=8, fontname="helv")
    p.insert_text((290, 830), str(number), fontsize=9, fontname="helv")
    return p


def schedule_page(doc, number):
    p = doc.new_page(width=595, height=842)
    p.insert_text((50, 60), "Periodic Maintenance Schedule", fontsize=15, fontname="hebo")
    p.insert_text((50, 85), "Carry out the following at the odometer reading shown, whichever comes first.",
                  fontsize=10, fontname="helv")
    p.insert_text((50, 100), "I = Inspect    R = Replace    C = Clean    L = Lubricate", fontsize=10, fontname="helv")
    rows = [["Item", "750", "3,000", "6,000", "9,000", "12,000"],
            ["Months", "1", "4", "8", "12", "16"],
            ["Engine oil", "R", "I", "R", "I", "R"],
            ["Oil filter", "R", "", "R", "", "R"],
            ["Spark plug", "", "I", "", "R", ""],
            ["Air filter element", "C", "C", "R", "C", "R"],
            ["Drive chain", "L", "L", "L", "L", "L"]]
    x0, y0, cw, rh = 50, 120, [150, 60, 60, 60, 60, 60], 24
    xs = [x0]
    for w in cw:
        xs.append(xs[-1] + w)
    for r in range(len(rows) + 1):
        p.draw_line((x0, y0 + r * rh), (xs[-1], y0 + r * rh))
    for x in xs:
        p.draw_line((x, y0), (x, y0 + len(rows) * rh))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell:
                p.insert_text((xs[c] + 5, y0 + r * rh + 16), cell, fontsize=10, fontname="helv")
    p.insert_text((50, 800), "Odometer readings are in km.", fontsize=9, fontname="helv")
    p.insert_text((50, 815), FOOTER, fontsize=8, fontname="helv")
    p.insert_text((290, 830), str(number), fontsize=9, fontname="helv")


def split_grid_schedule_page(doc, number):
    """Second schedule page mimicking real-world quirks (seen in a Honda manual): a finely split grid
    where each code sits in a sub-cell offset from its km heading, a '× 1,000 km' header, '–' empty
    cells, extra labelled columns, a note cell, and NO legend on this page (it is on page 5)."""
    p = doc.new_page(width=595, height=842)
    p.insert_text((50, 60), "Periodic Maintenance Schedule", fontsize=15, fontname="hebo")
    x_name, x_pre = 50, 200
    km_x = [260, 300, 340, 380]                   # each km column is 40 wide, split into two 20-wide sub-cells
    x_ref, x_end = 420, 480
    y0, rh = 90, 22
    rows = 5
    for r in range(rows + 1):
        p.draw_line((x_name, y0 + r * rh), (x_end, y0 + r * rh))
    for x in [x_name, x_pre, 260] + [k + 40 for k in km_x] + [x_end]:
        p.draw_line((x, y0), (x, y0 + rows * rh))
    for k in km_x:                                  # sub-cell split only in the data rows
        p.draw_line((k + 20, y0 + 2 * rh), (k + 20, y0 + rows * rh))
    p.insert_text((x_name + 4, y0 + 15), "Items", fontsize=9, fontname="helv")
    p.insert_text((x_pre + 4, y0 + 15), "Pre-ride Check", fontsize=8, fontname="helv")
    p.insert_text((x_ref + 4, y0 + 15), "Refer to page", fontsize=7, fontname="helv")
    p.insert_text((x_pre + 4, y0 + rh + 15), "× 1,000 km", fontsize=8, fontname="helv")
    for k, v in zip(km_x, ["1", "6", "12", "18"]):
        p.insert_text((k + 14, y0 + rh + 15), v, fontsize=9, fontname="helv")
    data = [("Brake shoes", "I", ["", "I", "", "I"], "–", ["L", "R", "R", "R"]),   # codes offset: right sub-cell
            ("Clutch cable", "I", ["I", "I", "I", "I"], "7", ["L", "L", "R", "L"]),
            ("Drive chain lube", "I", None, "5", None)]
    for n, (name, pre, codes, ref, side) in enumerate(data):
        y = y0 + (2 + n) * rh + 15
        p.insert_text((x_name + 4, y), name, fontsize=9, fontname="helv")
        p.insert_text((x_pre + 20, y), pre, fontsize=9, fontname="helv")
        p.insert_text((x_ref + 20, y), ref, fontsize=9, fontname="helv")
        if codes is None:
            p.insert_text((264, y), "500 km: I L", fontsize=8, fontname="helv")
        else:
            for k, c, sd in zip(km_x, codes, side):
                if c:
                    p.insert_text((k + (26 if sd == "R" else 6), y), c, fontsize=9, fontname="helv")
    p.insert_text((50, 815), FOOTER, fontsize=8, fontname="helv")
    p.insert_text((290, 830), str(number), fontsize=9, fontname="helv")


def build(path):
    doc = pymupdf.open()
    page(doc, 1, [("h", "Welcome"), ("b", "Thank you for choosing the Acme Roadster 150.\nRead this manual before riding.")])
    page(doc, 2, [("h", "Specifications"),
                  ("b", "Fuel tank capacity: 11 litres\nCurb weight: 142 kg\nSpark plug gap: 0.8 - 0.9 mm\nBattery: 12 V, 5 Ah\n"
                        "Spark plug\nCPR8EA-9 (NGK)\nBATTERY\nYTZ5S / GTZ5S"),
                  ("h", "Tyre Pressure"),
                  ("b", "Front: 25 psi\nRear: 29 psi (solo), 32 psi (with pillion)\nCheck tyre pressure when tyres are cold.")])
    page(doc, 3, [("h", "Checking Engine Oil"),
                  ("b", "Park the motorcycle on the centre stand on level ground.\n"
                        "Remove the dipstick, wipe it and reinsert it without screwing in.\n"
                        "The oil level should be between the upper and lower marks.\n"
                        "Recommended oil: SAE 10W-30 JASO MA."),
                  ("h", "Clutch Lever Play"),
                  ("b", "Clutch lever free play should be 10 - 20 mm at the lever tip.\n"
                        "Adjust using the adjuster at the clutch lever.")])
    page(doc, 4, [("h", "Trouble Shooting"),
                  ("b", "Problem: Engine does not start\nPossible cause: Fuel tank empty. Remedy: Refuel.\n"
                        "Possible cause: Engine stop switch in OFF position. Remedy: Set switch to RUN.\n"
                        "Possible cause: Battery discharged. Remedy: Charge the battery or contact a dealer.\n"
                        "Problem: Headlamp dim\nPossible cause: Weak battery. Remedy: Contact an authorised dealer.")])
    schedule_page(doc, 5)
    split_grid_schedule_page(doc, 6)
    doc.save(path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "other_manual.pdf")
