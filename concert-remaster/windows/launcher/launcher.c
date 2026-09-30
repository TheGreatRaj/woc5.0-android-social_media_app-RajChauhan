/*
 * Concert Remaster.exe - starts the app without a console window.
 *
 * Looks for the app next to itself (a "concert-remaster" folder, or this folder),
 * starts its private Python (pythonw.exe) with "-m concert_remaster gui", and
 * offers to run the one-time setup if it has not been done yet. If the app is
 * already open, the app itself brings the existing window to the front.
 */
#define WIN32_LEAN_AND_MEAN
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <shlwapi.h>
#include <wchar.h>

static BOOL exists(const wchar_t *path) { return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES; }

static BOOL find_app(wchar_t *app, size_t n) {
    wchar_t here[MAX_PATH], probe[MAX_PATH];
    GetModuleFileNameW(NULL, here, MAX_PATH);
    PathRemoveFileSpecW(here);
    const wchar_t *candidates[] = { L"\\concert-remaster", L"" };
    for (int i = 0; i < 2; i++) {
        swprintf(probe, MAX_PATH, L"%ls%ls\\pyproject.toml", here, candidates[i]);
        if (exists(probe)) {
            swprintf(app, n, L"%ls%ls", here, candidates[i]);
            return TRUE;
        }
    }
    return FALSE;
}

static BOOL start(wchar_t *cmd, const wchar_t *dir, DWORD flags) {
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, flags, NULL, dir, &si, &pi)) return FALSE;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return TRUE;
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE prev, PWSTR args, int show) {
    wchar_t app[MAX_PATH], python[MAX_PATH], cmd[4 * MAX_PATH];
    (void)inst; (void)prev; (void)show;
    if (!find_app(app, MAX_PATH)) {
        MessageBoxW(NULL, L"The Concert Remaster app folder was not found next to this program.\n\n"
                          L"Keep \"Concert Remaster.exe\" in the folder that contains \"concert-remaster\".",
                    L"Concert Remaster", MB_ICONERROR);
        return 1;
    }
    swprintf(python, MAX_PATH, L"%ls\\.venv\\Scripts\\pythonw.exe", app);
    if (!exists(python)) {
        int answer = MessageBoxW(NULL, L"Concert Remaster isn't set up on this PC yet (or setup didn't finish).\n\n"
                                       L"Run the setup now? It downloads Python, the AI libraries for your graphics card "
                                       L"and the AI models (about 20 GB in total) and then works offline.",
                                 L"Concert Remaster", MB_ICONQUESTION | MB_YESNO);
        if (answer != IDYES) return 1;
        swprintf(cmd, 4 * MAX_PATH,
                 L"powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File \"%ls\\windows\\setup.ps1\"", app);
        return start(cmd, app, CREATE_NEW_CONSOLE) ? 0 : 1;
    }
    swprintf(cmd, 4 * MAX_PATH, L"\"%ls\" -m concert_remaster gui %ls", python, args ? args : L"");
    if (!start(cmd, app, 0)) {
        MessageBoxW(NULL, L"Could not start Concert Remaster. Run setup again to repair it.", L"Concert Remaster", MB_ICONERROR);
        return 1;
    }
    return 0;
}
