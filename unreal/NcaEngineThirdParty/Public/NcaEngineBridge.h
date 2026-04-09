#pragma once

#include "CoreMinimal.h"

THIRD_PARTY_INCLUDES_START
#include "nca_engine.h"
THIRD_PARTY_INCLUDES_END

class FNcaEngineBridge
{
public:
    FNcaEngineBridge() = default;

    ~FNcaEngineBridge()
    {
        Destroy();
    }

    bool Create(const NcaEngineConfig& InConfig)
    {
        Destroy();

        NcaEngine* NewHandle = nullptr;
        const NcaStatus Status = nca_engine_create(&InConfig, &NewHandle);
        if (Status != NCA_STATUS_OK)
        {
            LastError = TEXT("nca_engine_create failed");
            return false;
        }

        Engine = NewHandle;
        return true;
    }

    void Destroy()
    {
        if (Engine != nullptr)
        {
            nca_engine_destroy(Engine);
            Engine = nullptr;
        }
    }

    bool Update(float DeltaSeconds)
    {
        return CallStatus(nca_engine_update(Engine, DeltaSeconds));
    }

    bool Render()
    {
        return CallStatus(nca_engine_render(Engine));
    }

    bool ResizeSurface(uint32 Width, uint32 Height)
    {
        return CallStatus(nca_engine_resize_surface(Engine, Width, Height));
    }

    bool SetStepsPerFrame(uint32 Steps)
    {
        return CallStatus(nca_engine_set_steps_per_frame(Engine, Steps));
    }

    bool LoadWeights(const FString& WeightPath)
    {
        FTCHARToUTF8 Utf8(*WeightPath);
        return CallStatus(nca_engine_load_weights(Engine, Utf8.Get()));
    }

    bool StartHotReload(const FString& WeightPath, uint32 DebounceMs = 150)
    {
        FTCHARToUTF8 Utf8(*WeightPath);
        return CallStatus(nca_engine_start_hot_reload(Engine, Utf8.Get(), DebounceMs));
    }

    bool StopHotReload()
    {
        return CallStatus(nca_engine_stop_hot_reload(Engine));
    }

    bool InjectDamage(float X, float Y, float Radius, float Strength, uint32 Frames = 1)
    {
        NcaBrushEvent Event;
        Event.x = X;
        Event.y = Y;
        Event.radius = Radius;
        Event.strength = Strength;
        Event.duration_frames = Frames;
        return CallStatus(nca_engine_inject_damage(Engine, &Event));
    }

    bool InjectGrowth(float X, float Y, float Radius, float Strength, uint32 Frames = 1)
    {
        NcaBrushEvent Event;
        Event.x = X;
        Event.y = Y;
        Event.radius = Radius;
        Event.strength = Strength;
        Event.duration_frames = Frames;
        return CallStatus(nca_engine_inject_growth(Engine, &Event));
    }

    const FString& GetLastError() const
    {
        return LastError;
    }

    NcaEngine* GetRawHandle() const
    {
        return Engine;
    }

private:
    bool CallStatus(NcaStatus Status)
    {
        if (Status == NCA_STATUS_OK)
        {
            LastError.Reset();
            return true;
        }

        if (Engine == nullptr)
        {
            LastError = TEXT("NCA engine handle is null");
            return false;
        }

        const size_t Needed = nca_engine_copy_last_error(Engine, nullptr, 0);
        if (Needed <= 1)
        {
            LastError = FString::Printf(TEXT("NCA call failed with status %d"), (int32)Status);
            return false;
        }

        TArray<char> Buffer;
        Buffer.SetNumZeroed((int32)Needed);
        nca_engine_copy_last_error(Engine, Buffer.GetData(), Buffer.Num());
        LastError = UTF8_TO_TCHAR(Buffer.GetData());
        return false;
    }

private:
    NcaEngine* Engine = nullptr;
    FString LastError;
};
