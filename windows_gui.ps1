param([switch]$SelfTest)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputRoot = Join-Path $projectRoot "output"
$startupLog = Join-Path $outputRoot "gui-startup.log"

try {
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $python = Join-Path $projectRoot ".venv-windows\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        throw "The Windows environment is missing. Run setup_windows.ps1 first."
    }

    $script:PipelineProcess = $null
    $script:CalibrationProcess = $null
    $script:PreviewProcess = $null
    $script:PublishProcess = $null
    $script:ModelSelectionProcess = $null
    $script:ReviewProcess = $null
    $script:ReviewForPipeline = $false
    $script:ReviewPromptShown = $false
    $script:ReviewControlPath = $null
    $script:PreviewResultPath = $null
    $script:ModelSelectionResultPath = $null
    $script:EventLogPath = $null
    $script:PipelineLogPath = $null
    $script:EventOffset = 0L
    $script:PipelineOffset = 0L
    $script:PipelineFinished = $true
    $script:CalibrationPath = $null
    $script:LastOutputDir = $null
    $script:WhistleModelPath = Join-Path $projectRoot "models\best_resnet_whistle_mix.pth"
    $script:RallyModelPath = Join-Path $projectRoot "models\rally_model_best.pth"
    $script:BallModelPath = Join-Path $projectRoot "models\temp_best_ball.pt"

    $colors = @{
        Background = [System.Drawing.Color]::FromArgb(11, 16, 32)
        Panel      = [System.Drawing.Color]::FromArgb(18, 26, 43)
        Input      = [System.Drawing.Color]::FromArgb(23, 33, 55)
        Border     = [System.Drawing.Color]::FromArgb(37, 50, 74)
        Text       = [System.Drawing.Color]::FromArgb(243, 246, 251)
        Muted      = [System.Drawing.Color]::FromArgb(148, 163, 184)
        Green      = [System.Drawing.Color]::FromArgb(34, 197, 94)
        Blue       = [System.Drawing.Color]::FromArgb(56, 189, 248)
        Yellow     = [System.Drawing.Color]::FromArgb(245, 158, 11)
        Red        = [System.Drawing.Color]::FromArgb(239, 68, 68)
    }

    function New-Label {
        param($Parent, [string]$Text, [int]$X, [int]$Y, [int]$Width, [int]$Height, [float]$Size = 9, [bool]$Bold = $false, $Color = $colors.Text)
        $label = New-Object System.Windows.Forms.Label
        $label.Text = $Text
        $label.Location = New-Object System.Drawing.Point($X, $Y)
        $label.Size = New-Object System.Drawing.Size($Width, $Height)
        $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
        $label.Font = New-Object System.Drawing.Font("Segoe UI", $Size, $style)
        $label.ForeColor = $Color
        $label.BackColor = $colors.Panel
        $Parent.Controls.Add($label)
        return $label
    }

    function Style-TextBox {
        param($Control)
        $Control.BackColor = $colors.Input
        $Control.ForeColor = $colors.Text
        $Control.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
        $Control.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    }

    function Style-Button {
        param($Control, [bool]$Primary = $false)
        $Control.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
        $Control.FlatAppearance.BorderSize = 0
        $Control.BackColor = if ($Primary) { $colors.Green } else { $colors.Input }
        $Control.ForeColor = if ($Primary) { [System.Drawing.Color]::FromArgb(7, 20, 11) } else { $colors.Text }
        $Control.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
        $Control.Cursor = [System.Windows.Forms.Cursors]::Hand
    }

    function Get-SafeMatchId {
        param([string]$Value)
        $clean = [regex]::Replace($Value, "[^A-Za-z0-9_-]+", "-").Trim('-', '_')
        if ([string]::IsNullOrWhiteSpace($clean)) { return "match" }
        return $clean
    }

    function Get-LoggedArgument {
        param([string]$LogPath, [string]$Name)
        if (-not $LogPath -or -not (Test-Path -LiteralPath $LogPath)) { return $null }
        $line = Get-Content -LiteralPath $LogPath -TotalCount 1 -ErrorAction SilentlyContinue
        if (-not $line) { return $null }
        $pattern = '(?:^|\s)' + [regex]::Escape($Name) + '\s+(?:"([^"]+)"|(\S+))'
        $match = [regex]::Match($line, $pattern)
        if (-not $match.Success) { return $null }
        if ($match.Groups[1].Success) { return $match.Groups[1].Value }
        return $match.Groups[2].Value
    }

    function Convert-TimeToSeconds {
        param([string]$Value)
        $parts = $Value.Trim().Split(':')
        if ($parts.Count -gt 3 -or $parts.Count -eq 0) { throw "Invalid timestamp: $Value" }
        [double]$seconds = 0
        foreach ($part in $parts) {
            [double]$number = 0
            if (-not [double]::TryParse($part, [ref]$number) -or $number -lt 0) {
                throw "Invalid timestamp: $Value"
            }
            $seconds = ($seconds * 60) + $number
        }
        return $seconds
    }

    function Convert-RangesToJson {
        param([string]$Text)
        $pieces = New-Object System.Collections.Generic.List[string]
        $lineNumber = 0
        foreach ($rawLine in ($Text -split "`r?`n")) {
            $line = $rawLine.Trim()
            if (-not $line) { continue }
            $lineNumber++
            $parts = $line -split '\s+-\s+', 2
            if ($parts.Count -ne 2) { throw "Set $lineNumber must use START - END." }
            $start = Convert-TimeToSeconds $parts[0]
            $end = Convert-TimeToSeconds $parts[1]
            if ($end -le $start) { throw "Set $lineNumber ends before it starts." }
            $pieces.Add("[$($start.ToString([System.Globalization.CultureInfo]::InvariantCulture)),$($end.ToString([System.Globalization.CultureInfo]::InvariantCulture))]")
        }
        if ($pieces.Count -eq 0) { return $null }
        return "[" + ($pieces -join ',') + "]"
    }

    function Join-ProcessArguments {
        param([string[]]$Values)
        return (($Values | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' ')
    }

    function Start-HiddenProcess {
        param([string]$FileName, [string[]]$Arguments)
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $FileName
        $info.Arguments = Join-ProcessArguments $Arguments
        $info.WorkingDirectory = $projectRoot
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        if (-not $process.Start()) { throw "Could not start $FileName" }
        return $process
    }

    function Read-NewText {
        param([string]$Path, [ref]$Offset)
        if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return "" }
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            if ($Offset.Value -gt $stream.Length) { $Offset.Value = 0L }
            [void]$stream.Seek($Offset.Value, [System.IO.SeekOrigin]::Begin)
            $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8, $true, 4096, $true)
            try { $text = $reader.ReadToEnd() } finally { $reader.Dispose() }
            $Offset.Value = $stream.Position
            return $text
        } finally {
            $stream.Dispose()
        }
    }

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Volleylitics - Match Processor"
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.ClientSize = New-Object System.Drawing.Size(1180, 820)
    $form.MinimumSize = New-Object System.Drawing.Size(1050, 760)
    $form.BackColor = $colors.Background
    $form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

    $brand = New-Object System.Windows.Forms.Label
    $brand.Text = "VOLLEYLITICS"
    $brand.Location = New-Object System.Drawing.Point(24, 13)
    $brand.Size = New-Object System.Drawing.Size(220, 20)
    $brand.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $brand.ForeColor = $colors.Green
    $brand.BackColor = $colors.Background
    $form.Controls.Add($brand)

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "Turn a match video into rally analytics"
    $title.Location = New-Object System.Drawing.Point(22, 32)
    $title.Size = New-Object System.Drawing.Size(700, 34)
    $title.Font = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
    $title.ForeColor = $colors.Text
    $title.BackColor = $colors.Background
    $form.Controls.Add($title)

    $left = New-Object System.Windows.Forms.Panel
    $left.Location = New-Object System.Drawing.Point(22, 76)
    $left.Size = New-Object System.Drawing.Size(370, 720)
    $left.BackColor = $colors.Panel
    $left.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left
    $form.Controls.Add($left)

    $right = New-Object System.Windows.Forms.Panel
    $right.Location = New-Object System.Drawing.Point(408, 76)
    $right.Size = New-Object System.Drawing.Size(748, 720)
    $right.BackColor = $colors.Panel
    $right.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
    $form.Controls.Add($right)

    New-Label $left "Match setup" 18 16 250 28 14 $true | Out-Null
    New-Label $left "VIDEO" 18 58 250 18 8 $true $colors.Muted | Out-Null
    $videoText = New-Object System.Windows.Forms.TextBox
    $videoText.Location = New-Object System.Drawing.Point(18, 80)
    $videoText.Size = New-Object System.Drawing.Size(254, 30)
    Style-TextBox $videoText
    $left.Controls.Add($videoText)
    $browseButton = New-Object System.Windows.Forms.Button
    $browseButton.Text = "Browse"
    $browseButton.Location = New-Object System.Drawing.Point(280, 79)
    $browseButton.Size = New-Object System.Drawing.Size(72, 29)
    Style-Button $browseButton
    $left.Controls.Add($browseButton)

    New-Label $left "MATCH NAME" 18 124 250 18 8 $true $colors.Muted | Out-Null
    $matchText = New-Object System.Windows.Forms.TextBox
    $matchText.Text = "match"
    $matchText.Location = New-Object System.Drawing.Point(18, 146)
    $matchText.Size = New-Object System.Drawing.Size(334, 30)
    Style-TextBox $matchText
    $left.Controls.Add($matchText)

    New-Label $left "SET RANGES - OPTIONAL" 18 190 250 18 8 $true $colors.Muted | Out-Null
    $rangesText = New-Object System.Windows.Forms.TextBox
    $rangesText.Multiline = $true
    $rangesText.ScrollBars = [System.Windows.Forms.ScrollBars]::Vertical
    $rangesText.Location = New-Object System.Drawing.Point(18, 212)
    $rangesText.Size = New-Object System.Drawing.Size(334, 88)
    Style-TextBox $rangesText
    $left.Controls.Add($rangesText)
    New-Label $left "One set per line: 00:05:10 - 00:29:08. Blank uses the full video." 18 304 334 30 8 $false $colors.Muted | Out-Null

    $previewButton = New-Object System.Windows.Forms.Button
    $previewButton.Text = "Preview video and mark sets"
    $previewButton.Location = New-Object System.Drawing.Point(18, 338)
    $previewButton.Size = New-Object System.Drawing.Size(334, 32)
    Style-Button $previewButton
    $left.Controls.Add($previewButton)

    $heatmapCheck = New-Object System.Windows.Forms.CheckBox
    $heatmapCheck.Text = "Generate ball tracking and court heatmap"
    $heatmapCheck.Checked = $true
    $heatmapCheck.Location = New-Object System.Drawing.Point(18, 382)
    $heatmapCheck.Size = New-Object System.Drawing.Size(330, 24)
    $heatmapCheck.ForeColor = $colors.Text
    $heatmapCheck.BackColor = $colors.Panel
    $left.Controls.Add($heatmapCheck)

    New-Label $left "COMPUTE DEVICE" 18 414 250 18 8 $true $colors.Muted | Out-Null
    $deviceChoice = New-Object System.Windows.Forms.ComboBox
    $deviceChoice.Location = New-Object System.Drawing.Point(18, 436)
    $deviceChoice.Size = New-Object System.Drawing.Size(334, 30)
    $deviceChoice.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
    $deviceChoice.BackColor = $colors.Input
    $deviceChoice.ForeColor = $colors.Text
    [void]$deviceChoice.Items.Add("GPU required - AMD ROCm or NVIDIA CUDA")
    [void]$deviceChoice.Items.Add("Automatic - GPU with CPU fallback")
    [void]$deviceChoice.Items.Add("CPU only")
    $deviceChoice.SelectedIndex = 0
    $left.Controls.Add($deviceChoice)

    New-Label $left "MODELS" 18 474 250 18 8 $true $colors.Muted | Out-Null
    $modelStatus = New-Label $left "Ball: temp_best_ball.pt - F1 94.80%, 196.77 FPS" 18 494 235 38 8 $false $colors.Muted
    $modelsButton = New-Object System.Windows.Forms.Button
    $modelsButton.Text = "Choose..."
    $modelsButton.Location = New-Object System.Drawing.Point(260, 494)
    $modelsButton.Size = New-Object System.Drawing.Size(92, 31)
    Style-Button $modelsButton
    $left.Controls.Add($modelsButton)

    $calibrationStatus = New-Label $left "Not calibrated" 18 539 190 25 9 $false $colors.Muted
    $calibrateButton = New-Object System.Windows.Forms.Button
    $calibrateButton.Text = "Calibrate court"
    $calibrateButton.Location = New-Object System.Drawing.Point(220, 534)
    $calibrateButton.Size = New-Object System.Drawing.Size(132, 31)
    Style-Button $calibrateButton
    $left.Controls.Add($calibrateButton)

    $runButton = New-Object System.Windows.Forms.Button
    $runButton.Text = "Run full pipeline"
    $runButton.Location = New-Object System.Drawing.Point(18, 577)
    $runButton.Size = New-Object System.Drawing.Size(334, 42)
    Style-Button $runButton $true
    $left.Controls.Add($runButton)

    $stopButton = New-Object System.Windows.Forms.Button
    $stopButton.Text = "Stop"
    $stopButton.Enabled = $false
    $stopButton.Location = New-Object System.Drawing.Point(18, 629)
    $stopButton.Size = New-Object System.Drawing.Size(96, 34)
    Style-Button $stopButton
    $left.Controls.Add($stopButton)
    $openButton = New-Object System.Windows.Forms.Button
    $openButton.Text = "Open results"
    $openButton.Enabled = $false
    $openButton.Location = New-Object System.Drawing.Point(120, 629)
    $openButton.Size = New-Object System.Drawing.Size(108, 34)
    Style-Button $openButton
    $left.Controls.Add($openButton)
    $publishButton = New-Object System.Windows.Forms.Button
    $publishButton.Text = "Add to site"
    $publishButton.Enabled = $false
    $publishButton.Location = New-Object System.Drawing.Point(234, 629)
    $publishButton.Size = New-Object System.Drawing.Size(118, 34)
    Style-Button $publishButton
    $left.Controls.Add($publishButton)
    $hintLabel = New-Label $left "Results are saved in output\<match-name>." 18 671 334 25 8 $false $colors.Muted

    New-Label $right "Processing stages" 18 16 300 28 14 $true | Out-Null
    $importButton = New-Object System.Windows.Forms.Button
    $importButton.Text = "Import analyzed"
    $importButton.Location = New-Object System.Drawing.Point(430, 14)
    $importButton.Size = New-Object System.Drawing.Size(130, 31)
    Style-Button $importButton
    $right.Controls.Add($importButton)
    $reviewButton = New-Object System.Windows.Forms.Button
    $reviewButton.Text = "Review ambiguous"
    $reviewButton.Enabled = $false
    $reviewButton.Location = New-Object System.Drawing.Point(570, 14)
    $reviewButton.Size = New-Object System.Drawing.Size(160, 31)
    Style-Button $reviewButton
    $right.Controls.Add($reviewButton)
    $stageList = New-Object System.Windows.Forms.ListView
    $stageList.Location = New-Object System.Drawing.Point(18, 52)
    $stageList.Size = New-Object System.Drawing.Size(712, 246)
    $stageList.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
    $stageList.View = [System.Windows.Forms.View]::Details
    $stageList.FullRowSelect = $true
    $stageList.GridLines = $false
    $stageList.HeaderStyle = [System.Windows.Forms.ColumnHeaderStyle]::Nonclickable
    $stageList.BackColor = $colors.Input
    $stageList.ForeColor = $colors.Text
    $stageList.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    [void]$stageList.Columns.Add("Stage", 185)
    [void]$stageList.Columns.Add("Status", 95)
    [void]$stageList.Columns.Add("Details", 410)
    $right.Controls.Add($stageList)

    $stageDefinitions = @(
        @("preflight", "1  Runtime check", "Python, FFmpeg, packages, and models"),
        @("calibration", "2  Court calibration", "Court corners and net-height boundary"),
        @("rallies", "3  Rally detection", "Whistles, rally model, and above-net ball activity"),
        @("review", "4  Decision review", "Automatic and uncertain intervals"),
        @("clips", "5  Rally clips", "One video clip per rally"),
        @("heatmap", "6  Ball analytics", "Landings, trajectories, and court map"),
        @("complete", "7  Results", "JSON data, clips, map, and processing log")
    )
    $script:StageItems = @{}
    foreach ($definition in $stageDefinitions) {
        $item = New-Object System.Windows.Forms.ListViewItem($definition[1])
        [void]$item.SubItems.Add("Waiting")
        [void]$item.SubItems.Add($definition[2])
        $item.ForeColor = $colors.Muted
        [void]$stageList.Items.Add($item)
        $script:StageItems[$definition[0]] = $item
    }

    $progress = New-Object System.Windows.Forms.ProgressBar
    $progress.Location = New-Object System.Drawing.Point(18, 312)
    $progress.Size = New-Object System.Drawing.Size(712, 8)
    $progress.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
    $right.Controls.Add($progress)

    $logBox = New-Object System.Windows.Forms.RichTextBox
    $logBox.Location = New-Object System.Drawing.Point(18, 334)
    $logBox.Size = New-Object System.Drawing.Size(712, 334)
    $logBox.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
    $logBox.BackColor = [System.Drawing.Color]::FromArgb(7, 11, 20)
    $logBox.ForeColor = [System.Drawing.Color]::FromArgb(203, 213, 225)
    $logBox.BorderStyle = [System.Windows.Forms.BorderStyle]::None
    $logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
    $logBox.ReadOnly = $true
    $right.Controls.Add($logBox)

    $summaryLabel = New-Label $right "Choose a match video to begin" 18 680 712 24 9 $false $colors.Muted
    $summaryLabel.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right

    function Set-StageStatus {
        param([string]$Stage, [string]$Status, [string]$Message = "")
        if (-not $script:StageItems.ContainsKey($Stage)) { return }
        $item = $script:StageItems[$Stage]
        switch ($Status) {
            "running"  { $label = "Running";  $color = $colors.Blue }
            "waiting"  { $label = "Review";   $color = $colors.Yellow }
            "paused"   { $label = "Paused";   $color = $colors.Yellow }
            "complete" { $label = "Complete"; $color = $colors.Green }
            "skipped"  { $label = "Skipped";  $color = $colors.Yellow }
            "failed"   { $label = "Failed";   $color = $colors.Red }
            default     { $label = "Waiting";  $color = $colors.Muted }
        }
        $item.SubItems[1].Text = $label
        if ($Message) { $item.SubItems[2].Text = $Message }
        $item.ForeColor = $color
    }

    function Reset-Stages {
        foreach ($definition in $stageDefinitions) { Set-StageStatus $definition[0] "pending" $definition[2] }
        if ($script:CalibrationPath -and (Test-Path -LiteralPath $script:CalibrationPath)) {
            Set-StageStatus "calibration" "complete" "Six court reference points saved"
        }
        if (-not $heatmapCheck.Checked) {
            Set-StageStatus "heatmap" "skipped" "Heatmap generation disabled"
        }
    }

    function Append-Log {
        param([string]$Text)
        if (-not $Text) { return }
        $logBox.AppendText($Text)
        if ($logBox.TextLength -gt 250000) {
            $logBox.Select(0, 50000)
            $logBox.SelectedText = ""
        }
        $logBox.SelectionStart = $logBox.TextLength
        $logBox.ScrollToCaret()
    }

    function Start-AmbiguousReview {
        param([bool]$ForPipeline = $false)
        if (-not $script:LastOutputDir) { throw "Import or process a match first." }
        if (-not (Test-Path -LiteralPath $videoText.Text)) { throw "Choose the original full-match video first." }
        foreach ($requiredName in @("hitl.json", "rallies.json")) {
            $requiredPath = Join-Path $script:LastOutputDir $requiredName
            if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Missing review file: $requiredPath" }
        }
        if ($script:ReviewProcess -and -not $script:ReviewProcess.HasExited) { return }
        $script:ReviewForPipeline = $ForPipeline
        $script:ReviewProcess = Start-HiddenProcess $python @(
            "-u", (Join-Path $projectRoot "rally_segmentator\hitl_reviewer.py"),
            "--match-id", (Get-SafeMatchId $matchText.Text),
            "--video", $videoText.Text,
            "--match-dir", $script:LastOutputDir
        )
        $reviewButton.Enabled = $false
        $summaryLabel.Text = "Ambiguous review open - Y rally, N not rally, S skip, Q quit"
    }

    function Refresh-Output {
        $eventText = Read-NewText $script:EventLogPath ([ref]$script:EventOffset)
        foreach ($line in ($eventText -split "`r?`n")) {
            if (-not $line.Trim()) { continue }
            try {
                $eventData = $line | ConvertFrom-Json
                Set-StageStatus $eventData.stage $eventData.status $eventData.message
                $summaryLabel.Text = $eventData.message
                if ($eventData.output_dir) { $script:LastOutputDir = $eventData.output_dir }
                if ($eventData.stage -eq "review" -and $eventData.status -eq "waiting" -and -not $script:ReviewPromptShown) {
                    $script:ReviewPromptShown = $true
                    $script:ReviewControlPath = $eventData.control_file
                    $answer = [System.Windows.Forms.MessageBox]::Show(
                        "$($eventData.hitl_count) ambiguous intervals were found.`r`n`r`nYes: review them now, then continue automatically.`r`nNo: continue without manual review.`r`nCancel: pause here and resume later.",
                        "Ambiguous rally review",
                        "YesNoCancel",
                        "Question"
                    )
                    if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
                        try {
                            Start-AmbiguousReview $true
                        } catch {
                            Set-Content -LiteralPath $script:ReviewControlPath -Value "pause" -Encoding ASCII
                            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot open review", "OK", "Error") | Out-Null
                        }
                    } elseif ($answer -eq [System.Windows.Forms.DialogResult]::No) {
                        Set-Content -LiteralPath $script:ReviewControlPath -Value "continue" -Encoding ASCII
                        $summaryLabel.Text = "Continuing without manual ambiguity review"
                    } else {
                        Set-Content -LiteralPath $script:ReviewControlPath -Value "pause" -Encoding ASCII
                        $summaryLabel.Text = "Pausing pipeline before clip generation"
                    }
                }
            } catch {
                Append-Log "[event] $line`r`n"
            }
        }
        $pipelineText = Read-NewText $script:PipelineLogPath ([ref]$script:PipelineOffset)
        if ($pipelineText) { Append-Log ($pipelineText -replace "`n", "`r`n") }
    }

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Choose a volleyball match video"
    $dialog.Filter = "Video files|*.mp4;*.mkv;*.mov;*.avi;*.m4v;*.webm|All files|*.*"
    $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderDialog.Description = "Choose an analyzed match output folder"
    $folderDialog.ShowNewFolderButton = $false

    $browseButton.Add_Click({
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
        $videoText.Text = $dialog.FileName
        $matchText.Text = Get-SafeMatchId ([System.IO.Path]::GetFileNameWithoutExtension($dialog.FileName))
        $script:LastOutputDir = Join-Path $outputRoot $matchText.Text
        $openButton.Enabled = Test-Path -LiteralPath $script:LastOutputDir
        $publishButton.Enabled = Test-Path -LiteralPath (Join-Path $script:LastOutputDir "heatmap\rally_results.json")
        $reviewButton.Enabled = (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "hitl.json")) -and
            (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "rallies.json"))
        $candidate = Join-Path $script:LastOutputDir "calibration.json"
        if (Test-Path -LiteralPath $candidate) {
            $script:CalibrationPath = $candidate
            $calibrationStatus.Text = "Ready - calibration.json"
            Set-StageStatus "calibration" "complete" "Six court reference points saved"
        } else {
            $script:CalibrationPath = $null
            $calibrationStatus.Text = "Not calibrated"
            Set-StageStatus "calibration" "pending" "Click Calibrate court before running"
        }
        $summaryLabel.Text = [System.IO.Path]::GetFileName($dialog.FileName)
    })

    $importButton.Add_Click({
        try {
            if ($script:PipelineProcess -and -not $script:PipelineProcess.HasExited) {
                throw "Stop the current processing run before importing another match."
            }
            if ($folderDialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
            $importedDir = [System.IO.Path]::GetFullPath($folderDialog.SelectedPath)
            $ralliesPath = Join-Path $importedDir "rallies.json"
            $clipsMetadataPath = Join-Path $importedDir "rallies_with_clips.json"
            $heatmapResultsPath = Join-Path $importedDir "heatmap\rally_results.json"
            if (-not (Test-Path -LiteralPath $ralliesPath) -and
                -not (Test-Path -LiteralPath $clipsMetadataPath) -and
                -not (Test-Path -LiteralPath $heatmapResultsPath)) {
                throw "This folder does not look like an analyzed match. Expected rallies.json, rallies_with_clips.json, or heatmap\rally_results.json."
            }

            $script:LastOutputDir = $importedDir
            $matchText.Text = Get-SafeMatchId (Split-Path -Leaf $importedDir)
            $script:PipelineLogPath = Join-Path $importedDir "pipeline.log"
            $script:PipelineOffset = 0L
            $rangesText.Clear()

            $statePath = Join-Path $importedDir "pipeline_state.json"
            $importedState = $null
            if (Test-Path -LiteralPath $statePath) {
                $importedState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
            }

            $loggedVideo = Get-LoggedArgument $script:PipelineLogPath "--video"
            if (-not $loggedVideo -and $importedState -and $importedState.video) {
                $loggedVideo = $importedState.video
            }
            if ($loggedVideo -and (Test-Path -LiteralPath $loggedVideo)) {
                $videoText.Text = $loggedVideo
            } else {
                $videoText.Clear()
                $answer = [System.Windows.Forms.MessageBox]::Show(
                    "The original video path was not found in this analysis folder. Choose the original full-match video now?",
                    "Original video needed for review",
                    "YesNo",
                    "Information"
                )
                if ($answer -eq [System.Windows.Forms.DialogResult]::Yes -and
                    $dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                    $videoText.Text = $dialog.FileName
                }
            }

            $loggedWhistleModel = Get-LoggedArgument $script:PipelineLogPath "--whistle-model"
            $loggedRallyModel = Get-LoggedArgument $script:PipelineLogPath "--rally-model"
            $loggedBallModel = Get-LoggedArgument $script:PipelineLogPath "--ball-model"
            if ($loggedWhistleModel -and (Test-Path -LiteralPath $loggedWhistleModel)) { $script:WhistleModelPath = $loggedWhistleModel }
            if ($loggedRallyModel -and (Test-Path -LiteralPath $loggedRallyModel)) { $script:RallyModelPath = $loggedRallyModel }
            if ($loggedBallModel -and (Test-Path -LiteralPath $loggedBallModel)) { $script:BallModelPath = $loggedBallModel }

            $calibrationCandidate = Join-Path $importedDir "calibration.json"
            if (Test-Path -LiteralPath $calibrationCandidate) {
                $script:CalibrationPath = $calibrationCandidate
                $calibrationStatus.Text = "Ready - calibration.json"
            } else {
                $script:CalibrationPath = $null
                $calibrationStatus.Text = "Calibration not found"
            }

            Reset-Stages
            if (Test-Path -LiteralPath $ralliesPath) { Set-StageStatus "rallies" "complete" "Imported rally timeline" }
            $hitlPath = Join-Path $importedDir "hitl.json"
            if (Test-Path -LiteralPath $hitlPath) { Set-StageStatus "review" "complete" "Uncertain intervals available for review" }
            if (Test-Path -LiteralPath (Join-Path $importedDir "rally_clips")) { Set-StageStatus "clips" "complete" "Imported rally clips" }
            if (Test-Path -LiteralPath $heatmapResultsPath) { Set-StageStatus "heatmap" "complete" "Imported ball analytics" }
            $canResumeHeatmap = (Test-Path -LiteralPath $clipsMetadataPath) -and
                (Test-Path -LiteralPath (Join-Path $importedDir "rally_clips")) -and
                -not (Test-Path -LiteralPath $heatmapResultsPath) -and
                ($importedState -and $importedState.phase -in @("failed", "heatmap_failed", "tracking_ball"))
            if ($importedState -and $importedState.phase -eq "awaiting_review") {
                Set-StageStatus "review" "waiting" "Paused before ambiguous review"
                Set-StageStatus "clips" "pending" "Will run after review/continue"
                Set-StageStatus "heatmap" "pending" "Will run after clip generation"
                Set-StageStatus "complete" "paused" "Ready to resume from saved rally detection"
                $runButton.Text = "Resume pipeline"
            } elseif ($canResumeHeatmap) {
                Set-StageStatus "rallies" "complete" "Reusing saved rally detection"
                Set-StageStatus "review" "complete" "Reusing saved review decisions"
                Set-StageStatus "clips" "complete" "Reusing existing rally clips"
                Set-StageStatus "heatmap" "failed" "Ball tracking stopped before completion"
                Set-StageStatus "complete" "paused" "Ready to resume only ball tracking"
                $runButton.Text = "Resume ball tracking"
            } else {
                Set-StageStatus "complete" "complete" "Analyzed match imported"
                $runButton.Text = "Run full pipeline"
            }

            $openButton.Enabled = $true
            $publishButton.Enabled = Test-Path -LiteralPath $heatmapResultsPath
            $reviewButton.Enabled = (Test-Path -LiteralPath $hitlPath) -and
                (Test-Path -LiteralPath $ralliesPath) -and
                (Test-Path -LiteralPath $videoText.Text)
            $logBox.Clear()
            Append-Log "Imported analyzed match: $importedDir`r`n"
            if ($videoText.Text) { Append-Log "Original video: $($videoText.Text)`r`n" }
            if (-not $reviewButton.Enabled -and (Test-Path -LiteralPath $hitlPath)) {
                Append-Log "Choose the original match video before reviewing ambiguous intervals.`r`n"
            }
            $summaryLabel.Text = if ($importedState -and $importedState.phase -eq "awaiting_review") {
                "Imported paused match - review ambiguous intervals or resume"
            } elseif ($canResumeHeatmap) {
                "Imported failed ball-tracking run - resume without repeating earlier stages"
            } else {
                "Imported $($matchText.Text)"
            }
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot import match", "OK", "Error") | Out-Null
        }
    })

    $reviewButton.Add_Click({
        try {
            Start-AmbiguousReview $false
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot review intervals", "OK", "Error") | Out-Null
        }
    })

    $heatmapCheck.Add_CheckedChanged({
        if ($heatmapCheck.Checked) {
            Set-StageStatus "heatmap" "pending" "Landings, trajectories, and court map"
        } else {
            Set-StageStatus "heatmap" "skipped" "Heatmap generation disabled"
        }
    })

    $previewButton.Add_Click({
        try {
            if (-not (Test-Path -LiteralPath $videoText.Text)) {
                throw "Choose a valid match video first."
            }
            if ($script:PreviewProcess -and -not $script:PreviewProcess.HasExited) { return }
            $initialRanges = Convert-RangesToJson $rangesText.Text
            if (-not $initialRanges) { $initialRanges = "[]" }
            $matchId = Get-SafeMatchId $matchText.Text
            $matchDir = Join-Path $outputRoot $matchId
            New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
            $previewStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
            $script:PreviewResultPath = Join-Path $matchDir ".preview-ranges-$previewStamp.txt"
            $script:PreviewProcess = Start-HiddenProcess $python @(
                "-u", (Join-Path $projectRoot "video_previewer.py"),
                "--video", $videoText.Text,
                "--output", $script:PreviewResultPath,
                "--initial-ranges-json", $initialRanges
            )
            $previewButton.Enabled = $false
            $summaryLabel.Text = "Video preview open - mark each set start and end"
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot open preview", "OK", "Error") | Out-Null
        }
    })

    $modelsButton.Add_Click({
        try {
            if ($script:ModelSelectionProcess -and -not $script:ModelSelectionProcess.HasExited) { return }
            $initialModels = @{
                whistle = $script:WhistleModelPath
                rally = $script:RallyModelPath
                ball = $script:BallModelPath
            } | ConvertTo-Json -Compress
            $modelStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
            $script:ModelSelectionResultPath = Join-Path $outputRoot ".model-selection-$modelStamp.json"
            $script:ModelSelectionProcess = Start-HiddenProcess $python @(
                "-u", (Join-Path $projectRoot "model_selector.py"),
                "--output", $script:ModelSelectionResultPath,
                "--initial-json", $initialModels
            )
            $modelsButton.Enabled = $false
            $summaryLabel.Text = "Model selection window open"
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot choose models", "OK", "Error") | Out-Null
        }
    })

    $calibrateButton.Add_Click({
        try {
            if (-not (Test-Path -LiteralPath $videoText.Text)) {
                [System.Windows.Forms.MessageBox]::Show("Choose a valid match video first.", "Video required", "OK", "Information") | Out-Null
                return
            }
            if ($script:CalibrationProcess -and -not $script:CalibrationProcess.HasExited) { return }
            $matchId = Get-SafeMatchId $matchText.Text
            $matchText.Text = $matchId
            $matchDir = Join-Path $outputRoot $matchId
            New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
            $script:CalibrationPath = Join-Path $matchDir "calibration.json"
            Set-StageStatus "calibration" "running" "Select the six marked court points in the calibration window"
            $calibrationStatus.Text = "Calibration window open"
            $script:CalibrationProcess = Start-HiddenProcess $python @(
                "-u", (Join-Path $projectRoot "heatmaps\point.py"),
                "--video", $videoText.Text,
                "--out", $script:CalibrationPath
            )
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Calibration failed", "OK", "Error") | Out-Null
            Set-StageStatus "calibration" "failed" $_.Exception.Message
        }
    })

    $runButton.Add_Click({
        try {
            if (-not (Test-Path -LiteralPath $videoText.Text)) { throw "Choose a valid match video first." }
            if ($script:PipelineProcess -and -not $script:PipelineProcess.HasExited) { return }
            if ($script:CalibrationProcess -and -not $script:CalibrationProcess.HasExited) { throw "Finish court calibration before starting processing." }
            $matchId = Get-SafeMatchId $matchText.Text
            $matchText.Text = $matchId
            $rangesJson = Convert-RangesToJson $rangesText.Text
            if (-not $script:CalibrationPath -or -not (Test-Path -LiteralPath $script:CalibrationPath)) {
                throw "Court calibration is required for the net-height ball boundary. Click Calibrate court and select all six points."
            }
            if ($script:ModelSelectionProcess -and -not $script:ModelSelectionProcess.HasExited) {
                throw "Finish choosing models before starting processing."
            }
            foreach ($modelPath in @($script:WhistleModelPath, $script:RallyModelPath, $script:BallModelPath)) {
                if (-not (Test-Path -LiteralPath $modelPath)) { throw "Selected model not found: $modelPath" }
            }

            $resumeAfterReview = $false
            $resumeHeatmap = $false
            if ($script:LastOutputDir) {
                $resumeStatePath = Join-Path $script:LastOutputDir "pipeline_state.json"
                if (Test-Path -LiteralPath $resumeStatePath) {
                    $resumeState = Get-Content -LiteralPath $resumeStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
                    $resumeAfterReview = $resumeState.phase -eq "awaiting_review"
                    $resumeHeatmap = $resumeState.phase -in @("heatmap_failed", "tracking_ball")
                    if ($resumeState.phase -eq "failed") {
                        $resumeHeatmap = (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "rallies_with_clips.json")) -and
                            (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "rally_clips")) -and
                            -not (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "heatmap\rally_results.json"))
                    }
                }
            }
            if ($resumeAfterReview -or $resumeHeatmap) {
                $matchDir = [System.IO.Path]::GetFullPath($script:LastOutputDir)
                $matchId = Get-SafeMatchId (Split-Path -Leaf $matchDir)
                $matchText.Text = $matchId
            } else {
                $matchDir = Join-Path $outputRoot $matchId
            }
            New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
            $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
            $script:EventLogPath = Join-Path $matchDir "events-$timestamp.jsonl"
            $script:PipelineLogPath = Join-Path $matchDir "pipeline.log"
            $script:ReviewControlPath = Join-Path $matchDir "review-control.txt"
            if (Test-Path -LiteralPath $script:ReviewControlPath) {
                Remove-Item -LiteralPath $script:ReviewControlPath -Force
            }
            $script:ReviewPromptShown = $false
            $script:EventOffset = 0L
            $script:PipelineOffset = 0L
            $script:LastOutputDir = $matchDir
            $script:PipelineFinished = $false
            $logBox.Clear()
            Append-Log $(if ($resumeAfterReview) { "Resuming after ambiguity review...`r`n" } elseif ($resumeHeatmap) { "Resuming ball tracking from saved clips...`r`n" } else { "Starting $([System.IO.Path]::GetFileName($videoText.Text))...`r`n" })
            Reset-Stages
            if ($resumeAfterReview) {
                Set-StageStatus "rallies" "complete" "Reusing saved rally detection"
                Set-StageStatus "review" "complete" "Using saved review decisions"
            } elseif ($resumeHeatmap) {
                Set-StageStatus "rallies" "complete" "Reusing saved rally detection"
                Set-StageStatus "review" "complete" "Using saved review decisions"
                Set-StageStatus "clips" "complete" "Reusing existing rally clips"
            }

            $arguments = New-Object System.Collections.Generic.List[string]
            foreach ($value in @(
                "-u", (Join-Path $projectRoot "run_pipeline.py"),
                "--video", $videoText.Text,
                "--match-id", $matchId,
                "--output-root", (Split-Path -Parent $matchDir),
                "--event-log", $script:EventLogPath
            )) { $arguments.Add($value) }
            if ($rangesJson) { $arguments.Add("--segments-json"); $arguments.Add($rangesJson) }
            $devicePolicy = @("gpu", "auto", "cpu")[$deviceChoice.SelectedIndex]
            $arguments.Add("--device")
            $arguments.Add($devicePolicy)
            $arguments.Add("--calibration")
            $arguments.Add($script:CalibrationPath)
            $arguments.Add("--whistle-model")
            $arguments.Add($script:WhistleModelPath)
            $arguments.Add("--rally-model")
            $arguments.Add($script:RallyModelPath)
            $arguments.Add("--ball-model")
            $arguments.Add($script:BallModelPath)
            $arguments.Add("--review-mode")
            $arguments.Add("prompt")
            $arguments.Add("--review-control")
            $arguments.Add($script:ReviewControlPath)
            if ($resumeAfterReview) { $arguments.Add("--resume-after-review") }
            if ($resumeHeatmap) { $arguments.Add("--resume-heatmap") }
            if (-not $heatmapCheck.Checked) { $arguments.Add("--skip-heatmap") }

            $script:PipelineProcess = Start-HiddenProcess $python $arguments.ToArray()
            $runButton.Enabled = $false
            $modelsButton.Enabled = $false
            $importButton.Enabled = $false
            $reviewButton.Enabled = $false
            $stopButton.Enabled = $true
            $openButton.Enabled = $false
            $progress.Style = [System.Windows.Forms.ProgressBarStyle]::Marquee
            $progress.MarqueeAnimationSpeed = 25
            $summaryLabel.Text = "Processing..."
            $runButton.Text = "Run full pipeline"
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot start processing", "OK", "Error") | Out-Null
        }
    })

    $stopButton.Add_Click({
        if ($script:PipelineProcess -and -not $script:PipelineProcess.HasExited) {
            & taskkill.exe /PID $script:PipelineProcess.Id /T /F 2>$null | Out-Null
            Set-StageStatus "complete" "failed" "Stopped by user"
            $summaryLabel.Text = "Processing stopped"
        }
    })

    $openButton.Add_Click({
        if ($script:LastOutputDir -and (Test-Path -LiteralPath $script:LastOutputDir)) {
            Start-Process explorer.exe -ArgumentList ('"' + $script:LastOutputDir + '"')
        }
    })

    $publishButton.Add_Click({
        try {
            $siteResults = if ($script:LastOutputDir) { Join-Path $script:LastOutputDir "heatmap\rally_results.json" } else { $null }
            if (-not $siteResults -or -not (Test-Path -LiteralPath $siteResults)) {
                throw "Run ball analytics first. The site needs heatmap\rally_results.json."
            }
            if ($script:PublishProcess -and -not $script:PublishProcess.HasExited) { return }
            $script:PublishProcess = Start-HiddenProcess $python @(
                "-u", (Join-Path $projectRoot "docs\add_match.py"),
                "--source-dir", $script:LastOutputDir
            )
            $publishButton.Enabled = $false
            $summaryLabel.Text = "Add match window open"
        } catch {
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot add match", "OK", "Error") | Out-Null
        }
    })

    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 350
    $timer.Add_Tick({
        if ($script:ReviewProcess -and $script:ReviewProcess.HasExited) {
            $reviewWasForPipeline = $script:ReviewForPipeline
            $reviewExitCode = $script:ReviewProcess.ExitCode
            $script:ReviewProcess.Dispose()
            $script:ReviewProcess = $null
            $hitlPath = if ($script:LastOutputDir) { Join-Path $script:LastOutputDir "hitl.json" } else { $null }
            $ralliesPath = if ($script:LastOutputDir) { Join-Path $script:LastOutputDir "rallies.json" } else { $null }
            $reviewButton.Enabled = -not $reviewWasForPipeline -and $hitlPath -and $ralliesPath -and
                (Test-Path -LiteralPath $hitlPath) -and
                (Test-Path -LiteralPath $ralliesPath) -and
                (Test-Path -LiteralPath $videoText.Text)
            if ($reviewExitCode -eq 0) {
                $decisionsPath = Join-Path $script:LastOutputDir "hitl_decisions.json"
                $decisionCount = 0
                if (Test-Path -LiteralPath $decisionsPath) {
                    $savedDecisions = Get-Content -LiteralPath $decisionsPath -Raw -Encoding UTF8 | ConvertFrom-Json
                    $decisionCount = @($savedDecisions).Count
                }
                if ($reviewWasForPipeline) {
                    $hitlCount = 0
                    if (Test-Path -LiteralPath $hitlPath) {
                        $allHitl = Get-Content -LiteralPath $hitlPath -Raw -Encoding UTF8 | ConvertFrom-Json
                        $hitlCount = @($allHitl).Count
                    }
                    $controlDecision = "reviewed"
                    if ($decisionCount -lt $hitlCount) {
                        $continuePartial = [System.Windows.Forms.MessageBox]::Show(
                            "You reviewed $decisionCount of $hitlCount ambiguous intervals. Continue the pipeline with these decisions?`r`n`r`nChoose No to pause and resume later.",
                            "Review incomplete",
                            "YesNo",
                            "Question"
                        )
                        if ($continuePartial -ne [System.Windows.Forms.DialogResult]::Yes) {
                            $controlDecision = "pause"
                        }
                    }
                    Set-Content -LiteralPath $script:ReviewControlPath -Value $controlDecision -Encoding ASCII
                    if ($controlDecision -eq "reviewed") {
                        Set-StageStatus "review" "running" "$decisionCount decisions saved; pipeline resuming"
                        $summaryLabel.Text = "Review integrated - resuming clip generation"
                    } else {
                        Set-StageStatus "review" "paused" "$decisionCount decisions saved; resume later"
                        $summaryLabel.Text = "Pipeline pausing with review progress saved"
                    }
                } else {
                    Set-StageStatus "review" "complete" "$decisionCount ambiguous decisions saved"
                    $summaryLabel.Text = "Review saved - accepted rallies are in rallies_with_hitl.json"
                }
            } else {
                if ($reviewWasForPipeline -and $script:ReviewControlPath) {
                    Set-Content -LiteralPath $script:ReviewControlPath -Value "pause" -Encoding ASCII
                    Set-StageStatus "review" "paused" "Reviewer failed; resume is still available"
                    $summaryLabel.Text = "Reviewer failed - pipeline will pause safely"
                } else {
                    Set-StageStatus "review" "failed" "Reviewer closed with an error"
                    $summaryLabel.Text = "Ambiguous review failed"
                }
            }
            $script:ReviewForPipeline = $false
        }

        if ($script:ModelSelectionProcess -and $script:ModelSelectionProcess.HasExited) {
            $modelExitCode = $script:ModelSelectionProcess.ExitCode
            $script:ModelSelectionProcess.Dispose()
            $script:ModelSelectionProcess = $null
            $modelsButton.Enabled = -not ($script:PipelineProcess -and -not $script:PipelineProcess.HasExited)
            if ($modelExitCode -eq 0 -and $script:ModelSelectionResultPath -and (Test-Path -LiteralPath $script:ModelSelectionResultPath)) {
                $selection = Get-Content -LiteralPath $script:ModelSelectionResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $script:WhistleModelPath = $selection.whistle
                $script:RallyModelPath = $selection.rally
                $script:BallModelPath = $selection.ball
                $ballName = [System.IO.Path]::GetFileName($script:BallModelPath)
                switch ($ballName) {
                    "yolo26m-960-best.pt" { $modelStatus.Text = "Ball: $ballName - F1 96.18%, 47.82 FPS" }
                    "temp_best_ball.pt" { $modelStatus.Text = "Ball: $ballName - F1 94.80%, 196.77 FPS" }
                    "best.pt" { $modelStatus.Text = "Ball: $ballName - F1 89.34%, 125.27 FPS" }
                    "last.pt" { $modelStatus.Text = "Ball: $ballName - F1 86.99%, 199.95 FPS" }
                    "best_old.pt" { $modelStatus.Text = "Ball: $ballName - F1 84.14%, 197.49 FPS" }
                    default { $modelStatus.Text = "Ball: $ballName - custom/no saved benchmark" }
                }
                Remove-Item -LiteralPath $script:ModelSelectionResultPath -Force
                $summaryLabel.Text = "Models updated for the next processing run"
            } elseif ($modelExitCode -ne 0) {
                $summaryLabel.Text = "Model selector could not open"
            }
            $script:ModelSelectionResultPath = $null
        }

        if ($script:PublishProcess -and $script:PublishProcess.HasExited) {
            $publishExitCode = $script:PublishProcess.ExitCode
            $script:PublishProcess.Dispose()
            $script:PublishProcess = $null
            $siteResults = if ($script:LastOutputDir) { Join-Path $script:LastOutputDir "heatmap\rally_results.json" } else { $null }
            $publishButton.Enabled = $siteResults -and (Test-Path -LiteralPath $siteResults)
            $summaryLabel.Text = if ($publishExitCode -eq 0) { "Add match window closed" } else { "Add match tool failed to open" }
        }

        if ($script:PreviewProcess -and $script:PreviewProcess.HasExited) {
            $previewExitCode = $script:PreviewProcess.ExitCode
            $script:PreviewProcess.Dispose()
            $script:PreviewProcess = $null
            $previewButton.Enabled = $true
            if ($previewExitCode -eq 0 -and $script:PreviewResultPath -and (Test-Path -LiteralPath $script:PreviewResultPath)) {
                $rangesText.Text = (Get-Content -LiteralPath $script:PreviewResultPath -Raw -Encoding UTF8).Trim()
                Remove-Item -LiteralPath $script:PreviewResultPath -Force
                $summaryLabel.Text = if ($rangesText.Text) { "Set ranges updated from video preview" } else { "Using the full video" }
            } elseif ($previewExitCode -ne 0) {
                $summaryLabel.Text = "Video preview could not open"
            }
            $script:PreviewResultPath = $null
        }

        if ($script:CalibrationProcess -and $script:CalibrationProcess.HasExited) {
            if ($script:CalibrationProcess.ExitCode -eq 0 -and (Test-Path -LiteralPath $script:CalibrationPath)) {
                $calibrationStatus.Text = "Ready - calibration.json"
                Set-StageStatus "calibration" "complete" "Six court reference points saved"
            } else {
                $calibrationStatus.Text = "Calibration not saved"
                Set-StageStatus "calibration" "failed" "Calibration was closed before six points were saved"
            }
            $script:CalibrationProcess.Dispose()
            $script:CalibrationProcess = $null
        }

        if ($script:PipelineProcess) {
            Refresh-Output
            if ($script:PipelineProcess.HasExited -and -not $script:PipelineFinished) {
                Refresh-Output
                $script:PipelineFinished = $true
                $exitCode = $script:PipelineProcess.ExitCode
                $pipelinePaused = $exitCode -eq 2
                $heatmapFailed = $false
                $statePath = Join-Path $script:LastOutputDir "pipeline_state.json"
                if (Test-Path -LiteralPath $statePath) {
                    try {
                        $currentState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
                        if ($currentState.phase -eq "awaiting_review") { $pipelinePaused = $true }
                        if ($currentState.phase -eq "heatmap_failed") { $heatmapFailed = $true }
                    } catch {}
                }
                $script:PipelineProcess.Dispose()
                $script:PipelineProcess = $null
                $runButton.Enabled = $true
                $modelsButton.Enabled = $true
                $importButton.Enabled = $true
                $stopButton.Enabled = $false
                $openButton.Enabled = Test-Path -LiteralPath $script:LastOutputDir
                $publishButton.Enabled = -not $pipelinePaused -and $exitCode -eq 0 -and (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "heatmap\rally_results.json"))
                $reviewButton.Enabled = ($pipelinePaused -or $exitCode -eq 0) -and
                    (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "hitl.json")) -and
                    (Test-Path -LiteralPath (Join-Path $script:LastOutputDir "rallies.json"))
                $progress.Style = [System.Windows.Forms.ProgressBarStyle]::Blocks
                $progress.Value = if ($pipelinePaused) { 45 } elseif ($heatmapFailed) { 80 } elseif ($exitCode -eq 0) { 100 } else { 0 }
                if ($pipelinePaused) {
                    $runButton.Text = "Resume pipeline"
                    Set-StageStatus "review" "paused" "Review now or resume without manual decisions"
                    Set-StageStatus "complete" "paused" "Detection saved; clips have not started"
                    $summaryLabel.Text = "Pipeline paused - review ambiguous intervals or resume later"
                }
                elseif ($heatmapFailed) {
                    $runButton.Text = "Resume ball tracking"
                    Set-StageStatus "heatmap" "failed" "Progress saved; a fresh GPU worker can resume"
                    Set-StageStatus "complete" "paused" "Earlier stages are complete"
                    $summaryLabel.Text = "Ball tracking stopped - resume from the saved checkpoint"
                }
                elseif ($exitCode -eq 0) {
                    $runButton.Text = "Run full pipeline"
                    $summaryLabel.Text = "Processing complete"
                }
                else {
                    $runButton.Text = "Run full pipeline"
                    $summaryLabel.Text = "Processing failed - see the log"
                    Set-StageStatus "complete" "failed" "Pipeline failed; see the log below"
                }
            }
        }
    })
    $timer.Start()

    $form.Add_FormClosing({
        param($sender, $eventArgs)
        if ($script:PipelineProcess -and -not $script:PipelineProcess.HasExited) {
            $answer = [System.Windows.Forms.MessageBox]::Show("Stop processing and close Volleylitics?", "Processing is running", "YesNo", "Warning")
            if ($answer -ne [System.Windows.Forms.DialogResult]::Yes) { $eventArgs.Cancel = $true; return }
            & taskkill.exe /PID $script:PipelineProcess.Id /T /F 2>$null | Out-Null
        }
        if ($script:CalibrationProcess -and -not $script:CalibrationProcess.HasExited) {
            & taskkill.exe /PID $script:CalibrationProcess.Id /T /F 2>$null | Out-Null
        }
        if ($script:PreviewProcess -and -not $script:PreviewProcess.HasExited) {
            & taskkill.exe /PID $script:PreviewProcess.Id /T /F 2>$null | Out-Null
        }
        if ($script:PublishProcess -and -not $script:PublishProcess.HasExited) {
            & taskkill.exe /PID $script:PublishProcess.Id /T /F 2>$null | Out-Null
        }
        if ($script:ModelSelectionProcess -and -not $script:ModelSelectionProcess.HasExited) {
            & taskkill.exe /PID $script:ModelSelectionProcess.Id /T /F 2>$null | Out-Null
        }
        if ($script:ReviewProcess -and -not $script:ReviewProcess.HasExited) {
            & taskkill.exe /PID $script:ReviewProcess.Id /T /F 2>$null | Out-Null
        }
    })

    "$(Get-Date -Format o) GUI starting" | Set-Content -LiteralPath $startupLog -Encoding UTF8
    if ($SelfTest) {
        $testRanges = Convert-RangesToJson "00:01:00 - 00:02:00`r`n03:00 - 04:00"
        if ($testRanges -ne "[[60,120],[180,240]]") {
            throw "Set-range self-test failed: $testRanges"
        }
        $form.CreateControl()
        Write-Output "Windows GUI self-test passed: $($form.Text)"
        $form.Dispose()
        exit 0
    }
    [void]$form.ShowDialog()
    exit 0
} catch {
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $details = "$(Get-Date -Format o)`r`n$($_ | Out-String)"
    $details | Set-Content -LiteralPath $startupLog -Encoding UTF8
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
        [System.Windows.Forms.MessageBox]::Show(
            "Volleylitics could not open.`r`n`r`n$($_.Exception.Message)`r`n`r`nDetails: $startupLog",
            "Volleylitics startup error",
            "OK",
            "Error"
        ) | Out-Null
    } catch {
        Write-Host $details
    }
    exit 1
}
