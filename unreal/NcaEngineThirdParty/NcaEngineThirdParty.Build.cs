using UnrealBuildTool;
using System.IO;

public class NcaEngineThirdParty : ModuleRules
{
    public NcaEngineThirdParty(ReadOnlyTargetRules Target) : base(Target)
    {
        Type = ModuleType.External;

        string ModulePath = ModuleDirectory;
        string IncludePath = Path.Combine(ModulePath, "include");
        PublicSystemIncludePaths.Add(IncludePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string LibPath = Path.Combine(ModulePath, "lib", "Win64");
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "nca_engine.lib"));

            PublicDelayLoadDLLs.Add("nca_engine.dll");
            RuntimeDependencies.Add(
                "$(PluginDir)/Source/ThirdParty/NcaEngineThirdParty/lib/Win64/nca_engine.dll"
            );
        }
    }
}
