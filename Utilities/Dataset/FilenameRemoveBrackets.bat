F:
cd F:\StableDiffusion\Datasets\2 Baking\Nusmusbim\New
setlocal enabledelayedexpansion
for %%a in (*.*) do (
set f=%%a
set f=!f:^(=!
set f=!f:^)=!
ren "%%a" "!f!"
)