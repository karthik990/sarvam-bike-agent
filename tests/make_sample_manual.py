"""Builds a tiny synthetic manual PDF for offline tests (not real OEM content)."""
import pymupdf, sys
pages = [
 "TROUBLESHOOTING\nExhaust Smoke\nWhite smoke from the exhaust when the engine is cold is normal condensation and will stop once the engine warms up. Persistent thick white smoke may indicate engine oil or coolant entering the combustion chamber. Check the engine oil level. If smoke persists, contact an authorised service centre.\nBlack smoke indicates a rich fuel mixture or a clogged air filter. Inspect the air filter element.",
 "ENGINE DOES NOT START\nCheck that the engine kill switch is in the RUN position. Ensure there is sufficient fuel in the tank. Check that the side stand is up and the transmission is in neutral. If the battery is weak, the starter motor will crank slowly; charge the battery.",
 "MAINTENANCE\nEngine Oil Level Check\nPark the motorcycle upright on level ground. Run the engine for 2 minutes and stop. Wait 1 minute. The oil level must be between the MIN and MAX marks on the dipstick. Use SAE 15W-50 oil.\nWARNING: Do not overfill.",
 "CLUTCH\nClutch lever free play should be 2-3 mm at the lever end. Adjust using the adjuster at the lever.",
]
doc = pymupdf.open()
for t in pages:
    p = doc.new_page(); p.insert_textbox(pymupdf.Rect(50,50,550,800), t, fontsize=11)
doc.save(sys.argv[1] if len(sys.argv)>1 else "sample_manual.pdf")
