import type { PIIProfile } from "../../types/piiProfile";

export type ProfilePatch = Partial<Pick<PIIProfile,
  "name" | "confidence" | "disabled_entities" | "custom_regexes">>;
type SaveResult = { ok: boolean; status?: number };
type Writer = (profile: PIIProfile) => Promise<{
  ok: boolean; status?: number; json(): Promise<PIIProfile>;
}>;

// Keep optimistic edits responsive, but send one full-profile PUT at a time.
// Rebase pending edits on the last confirmed value after any failed save.
export function createProfileSaveQueue(
  current: () => PIIProfile | null,
  apply: (profile: PIIProfile) => void,
  write: Writer
) {
  type Edit = { patch: ProfilePatch; resolve: (result: SaveResult) => void };
  type State = { confirmed: PIIProfile; edits: Edit[] };
  const states = new Map<string, State>();
  const publish = (id: string, state: State) => {
    if (current()?.id === id) {
      apply(state.edits.reduce((p, edit) => ({ ...p, ...edit.patch }), state.confirmed));
    }
  };
  const drain = async (id: string, state: State) => {
    while (state.edits.length) {
      const edit = state.edits[0];
      let result: SaveResult;
      try {
        const response = await write({ ...state.confirmed, ...edit.patch });
        result = { ok: response.ok, status: response.status };
        if (response.ok) state.confirmed = await response.json();
      } catch {
        result = { ok: false };
      }
      state.edits.shift();
      publish(id, state);
      edit.resolve(result);
    }
    states.delete(id);
  };
  return (patch: ProfilePatch): Promise<SaveResult> => {
    const profile = current();
    if (!profile) return Promise.resolve({ ok: false });
    let state = states.get(profile.id);
    const start = !state;
    if (!state) {
      state = { confirmed: profile, edits: [] };
      states.set(profile.id, state);
    }
    const pending = state;
    return new Promise(resolve => {
      pending.edits.push({ patch, resolve });
      publish(profile.id, pending);
      if (start) void drain(profile.id, pending);
    });
  };
}
