# Description: PowerShell script to run the Othello game in AI vs AI mode with different depths and heuristic combinations.
# Usage: Run the script in the PowerShell terminal to automate the game execution and log the results.

<#
.SYNOPSIS
Automates Othello AI vs AI matches with different parameters and detailed logging
#>

# Configuration
$logFile = "othello_v3_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$heuristicCombinations = @("1 2", "2 3", "3 1")
$maxDepth = 8
$stateFile = "othello_v3_automation_state.json"

# Create header for the log file
$logHeader = @"
============================================
Othello Automation Log
Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
System Info: PowerShell $($PSVersionTable.PSVersion)
Python Version: $(python --version 2>&1)
============================================

"@

# Only create new log file if we're not resuming
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
        [int]$CurrentDepth,
        [string]$CurrentHeuristics
    )
    
    $state = @{
        CurrentDepth = $CurrentDepth
        CurrentHeuristics = $CurrentHeuristics
        LogFile = $logFile
        LastUpdate = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    }
    
    $state | ConvertTo-Json | Out-File -FilePath $stateFile -Encoding utf8
    Write-Log "State saved: Depth=$CurrentDepth, Heuristics=$CurrentHeuristics" -Status "STATE"
}

function Get-SavedState {
    if (Test-Path $stateFile) {
        $state = Get-Content -Path $stateFile -Raw | ConvertFrom-Json
        Write-Log "Found saved state from $($state.LastUpdate)" -Status "RESUME"
        return $state
    }
    return $null
}

try {
    # Check for saved state
    $savedState = Get-SavedState
    $startDepth = 1
    $startHeuristicIndex = 0
    
    if ($savedState) {
        $logFile = $savedState.LogFile
        $startDepth = $savedState.CurrentDepth
        $startHeuristicIndex = $heuristicCombinations.IndexOf($savedState.CurrentHeuristics)
        
        if ($startHeuristicIndex -eq -1) {
            $startHeuristicIndex = 0
        } else {
            # Move to next combination as the previous one might have been interrupted
            $startHeuristicIndex++
            if ($startHeuristicIndex -ge $heuristicCombinations.Count) {
                $startHeuristicIndex = 0
                $startDepth++
            }
        }
        
        Write-Log "Resuming from Depth $startDepth, Heuristic combination index $startHeuristicIndex" -Status "RESUME"
    }

    # Main execution loop
    for ($depth = $startDepth; $depth -le $maxDepth; $depth++) {
        for ($i = $startHeuristicIndex; $i -lt $heuristicCombinations.Count; $i++) {
            $heuristics = $heuristicCombinations[$i]
            $heuristic1, $heuristic2 = $heuristics -split ' '
            $testId = "Depth-$depth-H$heuristic1-H$heuristic2"
            
            Write-Log "Starting test $testId"
            $startTime = Get-Date

            # Save current state before starting the test
            Save-State -CurrentDepth $depth -CurrentHeuristics $heuristics

            # Generate input sequence
            $inputToScript = @"
3
$depth
$depth
$heuristic1
$heuristic2
"@

            try {
                # Run the game with input redirection and capture output
                # Note: Modify the file name based on the version of the game you want to run
                # othello_v1 -> first version of the game
                # othello_v2 -> second version of the game
                # othello_v3 -> third version of the game
                $output = $inputToScript | python othello_v3.py 2>&1
                
                # Log results
                # ========================================================
                # Fix: Use proper formatting for duration calculation - 
                # Depths 9,8,10 are expected to take longer times to complete !!-- days, hours, minutes, seconds --!!
                # ========================================================
                # $ts = New-TimeSpan -Start $startTime -End (Get-Date)
                # $totalHours = $ts.Days * 24 + $ts.Hours
                # $duration = "{0}:{1:D2}:{2:D2}" -f $totalHours, $ts.Minutes, $ts.Seconds
                # ========================================================
                # Instead of using custom calculations, use built-in TimeSpan object
                # ========================================================
                $duration = (New-TimeSpan -Start $startTime -End (Get-Date)).ToString("dd\:hh\:mm\:ss")
                Write-Log "Completed $testId | Duration: $duration"
                
                # Fix: Use escape characters and proper variable expansion
                # Add-Content -Path $logFile -Value "`nGAME OUTPUT FOR ${testId}:`n$output`n" -Encoding utf8
            }
            catch {
                Write-Log "Error in $testId | $($_.Exception.Message)" -Status "ERROR"
                Write-Log "Stack Trace: $($_.ScriptStackTrace)" -Status "ERROR"
                throw  # Re-throw to trigger the finally block and maintain state
            }
            
            Write-Log "--------------------------------------------------"
        }
        # Reset heuristic index after completing a depth level
        $startHeuristicIndex = 0
    }

    # If we complete successfully, remove the state file
    if (Test-Path $stateFile) {
        Remove-Item $stateFile
        Write-Log "Completed all tests, removed state file" -Status "SUCCESS"
    }
}
finally {
    # Final log entry if we've completed all tests
    if ($depth -gt $maxDepth) {
        $completionMessage = @"


============================================
Automation Complete
Total Tests Run: $(($maxDepth * $heuristicCombinations.Count))
End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
============================================
"@
        Write-Log $completionMessage
    }
}