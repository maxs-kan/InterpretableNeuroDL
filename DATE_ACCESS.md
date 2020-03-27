Set-Location -Path ['Path to your database folder']
 dir | %{
 $id = $_.Name.SubString(0,6)
 ['Path to your WinRAR.exe']\WinRAR.exe e "['Path to your database folder']\$id*_3T_Structural_preproc.zip" -x! "$id*\T1w\T1w_acpc_dc_restore_brain.nii.gz" "['Create tmp folder and add it's path here']\tmp"
 Start-Sleep -s 5
 ['Path to your WinRAR.exe']\WinRAR.exe e ['Path to your tmp folder']\tmp\T1w_acpc_dc_restore_brain.nii.gz ['Create data folder and add it's path here']\data
 Start-Sleep -s 5
 Rename-Item -Path "['Path to your data folder']\data\T1w_acpc_dc_restore_brain.nii" -NewName "$id`.nii"
 Get-ChildItem -Path ['Path to your tmp folder']\tmp -Include *.* -Recurse | foreach { $_.Delete()}
 Start-Sleep -s 5
 }

dee

dede\de



de