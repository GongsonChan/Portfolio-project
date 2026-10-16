import update_meta_oof
import update_oof
import update_sequence_oof

if __name__ == "__main__":
    print("=== factor OOF ===", flush=True)
    update_oof.run()
    print("\n=== sequence OOF ===", flush=True)
    update_sequence_oof.run()
    print("\n=== meta OOF ===", flush=True)
    update_meta_oof.run()
    print("\ndone.", flush=True)
