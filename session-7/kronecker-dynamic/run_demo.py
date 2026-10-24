"""One command that runs the whole demonstration end to end.

    python run_demo.py

Runs the collision proofs (1 & 2) then trains both codecs (proofs 3 & 4),
regenerating collision_report.json, training_report.json, and the two charts
in ../assets/.
"""

import collision_test
import train

if __name__ == "__main__":
    print("\n########## COLLISION PROOFS (1 & 2) ##########\n")
    collision_test.main()
    print("\n########## TRAINING PROOFS (3 & 4) ##########\n")
    train.main()
    print("\nAll proofs complete. See collision_report.json, training_report.json, ../assets/*.png")
