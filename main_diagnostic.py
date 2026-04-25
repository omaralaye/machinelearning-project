import diagnostic_step1
import diagnostic_step2
import diagnostic_step3
import compare_results
import generate_report

def main():
    print("==========================================")
    print("   ELECTRICITY DEMAND FORECAST DIAGNOSTIC ")
    print("==========================================\n")

    # Step 1 & 2
    diagnostic_step1.diagnostic_step1()
    diagnostic_step2.diagnostic_step2()

    # Step 3
    diagnostic_step3.run_diagnostic_step3()

    # Step 4 & 5
    val_before, val_after = compare_results.run_comparison()

    # Step 6
    generate_report.generate_report(val_before, val_after)

    print("\nDiagnostic and Fix process complete.")
    print("Check 'diagnostic_comparison.png' and 'diagnostic_report.txt' for details.")

if __name__ == "__main__":
    main()
