# Description: PowerShell script to run OthelloGameEngine in AI vs AI mode with all depth combinations (1-8 vs 1-8)
# Usage: Run in PowerShell to automate AI battles and log results with execution details

<#
.SYNOPSIS
Automates Othello AI vs AI matches with all depth combinations (1-8) and detailed logging
#>

# Configuration
$logFile = "OthelloAI_Log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$depthCombinations = @()
foreach ($d1 in 1..8) {
    foreach ($d2 in 1..8) {
        $depthCombinations += "$d1 $d2"
    }
}
$stateFile = "othello_engine_state.json"

# Create header for the log file
$logHeader = @"
============================================
Othello AI Benchmark Log
Core Engine: OthelloGameEngine.py
Advanced Features: Pattern Matching, Mobility Analysis, Stability Evaluation
Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Total Tests: $($depthCombinations.Count)
Depth Range: 1-8 for both AI players
============================================

"@

if (-not (Test-Path $stateFile)) {
    $logHeader | Out-File -FilePath $logFile -Encoding utf8
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Status = "INFO"
    )
    
    $logEntry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Status] $Message"
    Add-Content -Path $logFile -Value $logEntry -Encoding utf8
    Write-Host $logEntry
}

function Save-State {
    param(
        [int]$CurrentTestIndex
    )
    
    $state = @{
        CurrentTestIndex = $CurrentTestIndex
        LogFile = $logFile
        LastUpdate = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    }
    
    $state | ConvertTo-Json | Out-File -FilePath $stateFile -Encoding utf8
    Write-Log "State saved: TestIndex=$CurrentTestIndex" -Status "STATE"
}

function Get-SavedState {
    if (Test-Path $stateFile) {
        $state = Get-Content -Path $stateFile -Raw | ConvertFrom-Json
        Write-Log "Resuming from saved state: Test #$($state.CurrentTestIndex)" -Status "RESUME"
        return $state
    }
    return $null
}

try {
    $savedState = Get-SavedState
    $startIndex = if ($savedState) { $savedState.CurrentTestIndex } else { 0 }

    # Main test loop
    for ($i = $startIndex; $i -lt $depthCombinations.Count; $i++) {
        $depths = $depthCombinations[$i] -split ' '
        $depth1, $depth2 = $depths[0], $depths[1]
        $testId = "AI1-d${depth1}_vs_AI2-d${depth2}"
        
        Write-Log "Starting test $testId (Test #$($i+1)/$($depthCombinations.Count))"
        Save-State -CurrentTestIndex $i
        $startTime = Get-Date

        # Generate input sequence for the game
        $inputToScript = @"
3
$depth1
$depth2
"@

        try {
            # Execute game with input redirection
            $output = $inputToScript | python OthelloGameEngine.py 2>&1
            
            # Parse and log results
            $finalScore = $output | Select-String "Final score - Black: (\d+), White: (\d+)"
            $winner = if ($finalScore.Matches.Groups[1].Value -gt $finalScore.Matches.Groups[2].Value) {
                "Black (AI1-d$depth1)"
            } elseif ($finalScore.Matches.Groups[1].Value -lt $finalScore.Matches.Groups[2].Value) {
                "White (AI2-d$depth2)"
            } else { "Draw" }

            $duration = (New-TimeSpan -Start $startTime -End (Get-Date)).ToString("dd\:hh\:mm\:ss")
            Write-Log "Completed $testId | Duration: $duration | Result: $winner | $($finalScore.Matches[0].Value)"
            
            # Save raw output
            # Add-Content -Path $logFile -Value "`nGAME OUTPUT FOR ${testId}:`n$output`n" -Encoding utf8
        }
        catch {
            Write-Log "Error in $testId | $($_.Exception.Message)" -Status "ERROR"
            throw
        }
        
        Write-Log "--------------------------------------------------"
    }

    # Cleanup state file after completion
    if (Test-Path $stateFile) {
        Remove-Item $stateFile
        Write-Log "All tests completed successfully" -Status "SUCCESS"
    }
}
finally {
    # Final summary

    # Extract Start Time dynamically from log file instead of substring indexing
    # Read full log content
    $logContent = Get-Content -Path $logFile -Raw
    # Use regex to find Start Time line and extract timestamp
    $startTimeMatch = $logContent | Select-String -Pattern "Start Time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

    # Ensure a match was found before processing
    if ($startTimeMatch) {
        # Convert extracted string to DateTime
        $startTime = [DateTime]::Parse($startTimeMatch.Matches.Groups[1].Value)
        # Calculate time span properly
        $totalDuration = New-TimeSpan -Start $startTime -End (Get-Date)

        $completionMessage = @"

============================================
Benchmark Complete
Total Matches: $($depthCombinations.Count)
Total Execution Time: $($totalDuration.ToString("dd\:hh\:mm\:ss"))
End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
============================================
"@
        Add-Content -Path $logFile -Value $completionMessage
    } else {
        # Handle missing start time gracefully
        Write-Log "Start Time not found in log. Time calculation skipped." -Status "WARNING"
    }
}
