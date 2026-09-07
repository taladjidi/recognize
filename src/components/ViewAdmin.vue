<!--
  - Copyright (c) 2021. The Recognize contributors.
  -
  - This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
  -->

<template>
	<div id="recognize">
		<figure v-if="loading" class="icon-loading loading" />
		<figure v-if="!loading && success" class="icon-checkmark success" />
		<NcSettingsSection :name="t('recognize', 'Status')">
			<NcNoteCard v-if="modelsDownloaded" show-alert type="success">
				{{ t('recognize', 'The machine learning models have been downloaded successfully.') }}
			</NcNoteCard>
			<NcNoteCard v-else-if="!modelsDownloaded" type="warning">
				{{ t('recognize', 'The machine learning models still need to be downloaded.') }}
			</NcNoteCard>
			<NcNoteCard v-if="tagsEnabled" show-alert type="success">
				{{ t('recognize', 'The systemtags app is enabled.') }}
			</NcNoteCard>
			<NcNoteCard v-else-if="!tagsEnabled" type="warning">
				{{ t('recognize', 'The systemtags app is currently disabled. Some features of this app will not work.') }}
			</NcNoteCard>
			<NcNoteCard v-if="daemon && !daemon.healthy" type="error">
				{{ t('recognize', 'The recognize background service is not running. Files will not be classified until it is started.') }}
			</NcNoteCard>
			<NcNoteCard v-if="cron !== undefined && cron !== 'cron'" type="error">
				{{ t('recognize', 'Background Jobs are not executed via cron. Recognize requires background jobs to be executed via cron.') }}
			</NcNoteCard>
			<template v-if="cron === 'cron'">
				<template v-if="settings['faces.enabled'] || settings['imagenet.enabled'] || settings['musicnn.enabled'] || settings['movinet.enabled']">
					<NcNoteCard show-alert type="success">
						{{ t('recognize', 'The app is installed and will automatically classify files in background processes.') }}
					</NcNoteCard>
				</template>
				<p v-else>
					{{ t('recognize', 'None of the tagging options below are currently selected. The app will currently do nothing.') }}
				</p>
			</template>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Face recognition')">
			<template v-if="settings['faces.enabled']">
				<NcNoteCard v-if="settings['faces.status'] === true" show-alert type="success">
					{{ t('recognize', 'Face recognition is working. ') }}
				</NcNoteCard>
				<NcNoteCard v-else-if="settings['faces.status'] === false" show-alert type="error">
					{{ t('recognize', 'An error occurred during face recognition, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-else type="warning">
					{{ t('recognize', 'Waiting for status reports on face recognition. If this message persists beyond 15 minutes, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Face recognition:') }} {{ countQueued.faces }} {{ t('recognize', 'Queued files') }}, {{ t('recognize', 'Last classification: ') }} {{ showDate(settings['faces.lastFile']) }}<span v-if="facesJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ facesJobs.scheduled }}, {{ facesJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(facesJobs.lastRun) : '' }}</span>
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Face clustering:') }} {{ countQueued.clusterFaces }} {{ t('recognize', 'faces left to cluster') }}, {{ t('recognize', 'Last clustering run: ') }} {{ showDate(settings['clusterFaces.lastRun']) }}<span v-if="clusterFacesJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ clusterFacesJobs.scheduled }}, {{ clusterFacesJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(clusterFacesJobs.lastRun) : '' }}</span><br>
					<small>{{ t('recognize', 'A minimum of 120 faces per user is necessary for clustering to kick in') }}</small>
				</NcNoteCard>
			</template>
			<p>
				<NcCheckboxRadioSwitch :checked.sync="settings['faces.enabled']" type="switch" @update:checked="onChange">
					{{ t('recognize', 'Enable face recognition (groups photos by faces that appear in them; UI is in the photos app)') }}
				</NcCheckboxRadioSwitch>
				<NcTextField :disabled="!settings['faces.enabled']"
					:value.sync="settings['faces.batchSize']"
					:label-visible="true"
					:label="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~500 or more, in WASM mode ~50 is recommended)')"
					:title="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~500 or more, in WASM mode ~50 is recommended)')"
					@update:value="onChange" />
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Object detection & landmark recognition')">
			<template v-if="settings['imagenet.enabled']">
				<NcNoteCard v-if="settings['imagenet.status'] === true" show-alert type="success">
					{{ t('recognize', 'Object recognition is working.') }}
				</NcNoteCard>
				<NcNoteCard v-else-if="settings['imagenet.status'] === false" show-alert type="error">
					{{ t('recognize', 'An error occurred during object recognition, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-else type="warning">
					{{ t('recognize', 'Waiting for status reports on object recognition. If this message persists beyond 15 minutes, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Object recognition:') }} {{ countQueued.imagenet }} {{ t('recognize', 'Queued files') }}, {{ t('recognize', 'Last classification: ') }} {{ showDate(settings['imagenet.lastFile']) }}<span v-if="imagenetJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ imagenetJobs.scheduled }}, {{ imagenetJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(imagenetJobs.lastRun) : '' }}</span>
				</NcNoteCard>
			</template>
			<template v-if="settings['landmarks.enabled']">
				<NcNoteCard v-if="settings['landmarks.status'] === true" show-alert type="success">
					{{ t('recognize', 'Landmark recognition is working.') }}
				</NcNoteCard>
				<NcNoteCard v-else-if="settings['landmarks.status'] === false" show-alert type="error">
					{{ t('recognize', 'An error occurred during landmark recognition, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-else type="warning">
					{{ t('recognize', 'Waiting for status reports on landmark recognition. If this message persists beyond 15 minutes, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Landmark recognition:') }} {{ countQueued.landmarks }} {{ t('recognize', 'Queued files') }}, {{ t('recognize', 'Last classification: ') }} {{ showDate(settings['landmarks.lastFile']) }}<span v-if="landmarksJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ landmarksJobs.scheduled }}, {{ landmarksJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(landmarksJobs.lastRun) : '' }}</span>
				</NcNoteCard>
			</template>

			<p>
				<NcCheckboxRadioSwitch :checked.sync="settings['imagenet.enabled']" type="switch" @update:checked="onChange">
					{{ t('recognize', 'Enable object recognition (e.g. food, vehicles, landscapes)') }}
				</NcCheckboxRadioSwitch>
				<NcTextField :disabled="!settings['imagenet.enabled']"
					:value.sync="settings['imagenet.batchSize']"
					:label-visible="true"
					:label="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					:title="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					@update:value="onChange" />
			</p>
			<p>&nbsp;</p>
			<p>
				<NcCheckboxRadioSwitch :checked.sync="settings['landmarks.enabled']"
					type="switch"
					:disabled="!settings['imagenet.enabled']"
					@update:checked="onChange">
					{{ t('recognize', 'Enable landmark recognition (e.g. Eiffel Tower, Golden Gate Bridge)') }}
				</NcCheckboxRadioSwitch>
				<NcTextField :disabled="!settings['imagenet.enabled'] || !settings['landmarks.enabled']"
					:value.sync="settings['landmarks.batchSize']"
					:label-visible="true"
					:label="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					:title="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					@update:value="onChange" />
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Audio tagging')">
			<template v-if="settings['musicnn.enabled']">
				<NcNoteCard v-if="settings['musicnn.status'] === true" show-alert type="success">
					{{ t('recognize', 'Audio recognition is working.') }}
				</NcNoteCard>
				<NcNoteCard v-else-if="settings['musicnn.status'] === false" show-alert type="error">
					{{ t('recognize', 'An error occurred during audio recognition, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-else type="warning">
					{{ t('recognize', 'Waiting for status reports on audio recognition. If this message persists beyond 15 minutes, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Music genre recognition:') }} {{ countQueued.musicnn }} {{ t('recognize', 'Queued files') }}, {{ t('recognize', 'Last classification: ') }} {{ showDate(settings['musicnn.lastFile']) }}<span v-if="musicnnJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ musicnnJobs.scheduled }}, {{ musicnnJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(musicnnJobs.lastRun) : '' }}</span>
				</NcNoteCard>
			</template>
			<p>
				<NcCheckboxRadioSwitch :checked.sync="settings['musicnn.enabled']" type="switch" @update:checked="onChange">
					{{ t('recognize', 'Enable music genre recognition (e.g. pop, rock, folk, metal, new age)') }}
				</NcCheckboxRadioSwitch>
				<NcTextField :disabled="!settings['musicnn.enabled']"
					:value.sync="settings['musicnn.batchSize']"
					:label-visible="true"
					:label="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					:title="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~100 or more, in WASM mode ~20 is recommended)')"
					@update:value="onChange" />
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Video tagging')">
			<template v-if="settings['movinet.enabled']">
				<NcNoteCard v-if="settings['movinet.status'] === true" show-alert type="success">
					{{ t('recognize', 'Video recognition is working.') }}
				</NcNoteCard>
				<NcNoteCard v-else-if="settings['movinet.status'] === false" show-alert type="error">
					{{ t('recognize', 'An error occurred during video recognition, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-else type="warning">
					{{ t('recognize', 'Waiting for status reports on video recognition. If this message persists beyond 15 minutes, please check the Nextcloud logs.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countQueued" type="info">
					{{ t('recognize', 'Video recognition:') }} {{ countQueued.movinet }} {{ t('recognize', 'Queued files') }}, {{ t('recognize', 'Last classification: ') }} {{ showDate(settings['movinet.lastFile']) }}<span v-if="movinetJobs">, {{ t('recognize', 'Scheduled background jobs: ') }} {{ movinetJobs.scheduled }}, {{ movinetJobs.lastRun ? t('recognize', 'Last background job execution: ') + showDate(movinetJobs.lastRun) : '' }}</span>
				</NcNoteCard>
			</template>
			<p>
				<NcCheckboxRadioSwitch :checked.sync="settings['movinet.enabled']"
					type="switch"
						@update:checked="onChange">
					{{ t('recognize', 'Enable human action recognition (e.g. arm wrestling, dribbling basketball, hula hooping)') }}
				</NcCheckboxRadioSwitch>
				<NcTextField :disabled="!settings['movinet.enabled']"
					:value.sync="settings['movinet.batchSize']"
					:label-visible="true"
					:label="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~20 or more, in WASM mode ~5 is recommended)')"
					:title="t('recognize', 'The number of files to process per job run (A job will be scheduled every 5 minutes; For normal operation ~20 or more, in WASM mode ~5 is recommended)')"
					@update:value="onChange" />
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Reset')">
			<p>{{ t('recognize', 'Click the button below to remove all tags from all files that have been classified so far.') }}</p>
			<button class="button" @click="onReset">
				{{ t('recognize', 'Reset tags for classified files') }}
			</button>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'Click the button below to remove all face detections from all files that have been classified so far.') }}</p>
			<button class="button" @click="onResetFaces">
				{{ t('recognize', 'Reset faces for classified files') }}
			</button>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'Click the button below to rescan all files in this instance and add them to the classifier queues.') }}</p>
			<button class="button" @click="onRescan">
				{{ t('recognize', 'Rescan all files') }}
			</button>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'Click the button below to clear the classifier queues and clear all background jobs. This is useful when you want to do the initial classification using the terminal command.') }}</p>
			<button class="button" @click="onClearJobs">
				{{ t('recognize', 'Clear queues and background jobs') }}
			</button>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Background service')">
			<p v-if="daemon === undefined">
				<span class="icon-loading-small" />&nbsp;&nbsp;&nbsp;&nbsp;{{ t('recognize', 'Checking the recognize service') }}
			</p>
			<template v-else>
				<NcNoteCard v-if="daemon.healthy" type="success">
					{{ t('recognize', 'The recognize service is running.') }}
					<span v-if="daemon.lastSeen">{{ t('recognize', 'Last seen: ') }}{{ showDate(daemon.lastSeen) }}</span>
				</NcNoteCard>
				<NcNoteCard v-else type="error">
					{{ t('recognize', 'The recognize service is not running, so no files are being classified.') }}
					<span v-if="daemon.lastSeen">{{ t('recognize', 'Last seen: ') }}{{ showDate(daemon.lastSeen) }}</span>
					<span v-else>{{ t('recognize', 'It has never reported in.') }}</span>
				</NcNoteCard>
				<p v-if="daemon.pending !== undefined">
					{{ daemon.pending }} {{ t('recognize', 'files waiting to be processed') }}
				</p>
			</template>
			<p>
				{{ t('recognize', 'Classification runs in a separate Python service using ONNX Runtime. It is managed outside Nextcloud (systemd unit "recognize"), and reports here on every poll.') }}
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'FFMPEG')">
			<p v-if="ffmpeg === undefined">
				<span class="icon-loading-small" />&nbsp;&nbsp;&nbsp;&nbsp;{{ t('recognize', 'Checking FFmpeg') }}
			</p>
			<NcNoteCard v-else-if="ffmpeg === false" type="warning">
				{{ t('recognize', 'Could not execute the FFmpeg binary. You may need to set the path to a working binary manually.') }}
			</NcNoteCard>
			<NcNoteCard v-else type="success">
				{{ t('recognize', 'FFmpeg {version} binary was installed successfully.', { version: ffmpeg }) }}
			</NcNoteCard>
			<p>
				{{ t('recognize', 'If the shipped FFmpeg binary doesn\'t work on your system for some reason you can set the path to a custom FFmpeg binary.') }}
			</p>
			<p>
				<NcTextField :value.sync="settings['ffmpeg_binary']" @update:value="onChange" />
			</p>
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Terminal commands') ">
			<p>{{ t('recognize', 'To queue every existing file for classification, run the following on the server terminal. The recognize background service picks the work up from there.') }}</p>
			<pre><code>occ recognize:recrawl</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'Before running a full initial classification run on the terminal, you should stop all background processing that Recognize scheduled upon installation to avoid interference.') }}</p>
			<pre><code>occ recognize:clear-background-jobs</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'To run a face clustering run on for each user in the terminal, run the following. Consider adding the parameter --batch-size 10000 for large libraries to avoid PHP memory exhaustion. (The clustering will run in sequence inside your terminal.)') }}</p>
			<pre><code>occ recognize:cluster-faces</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'To remove all face clusters but keep the raw detected faces run the following on the terminal:') }}</p>
			<pre><code>occ recognize:reset-face-clusters</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'To remove all detected faces and face clusters run the following on the terminal:') }}</p>
			<pre><code>occ recognize:reset-faces</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'You can reset the tags of all files that have been previously classified by Recognize with the following command:') }}</p>
			<pre><code>occ recognize:reset-tags</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'You can delete all tags that no longer have any files associated with them with the following command:') }}</p>
			<pre><code>occ recognize:cleanup-tags</code></pre>
			<p>&nbsp;</p>
			<p>{{ t('recognize', 'To remove tags that were created by Recognize version 2 from all files run the following on the terminal:') }}</p>
			<pre><code>occ recognize:remove-legacy-tags</code></pre>
		</NcSettingsSection>
	</div>
</template>

<script>
import { NcNoteCard, NcSettingsSection, NcCheckboxRadioSwitch, NcTextField } from '@nextcloud/vue'
import axios from '@nextcloud/axios'
import { generateUrl } from '@nextcloud/router'
import { loadState } from '@nextcloud/initial-state'
import humanizeDuration from 'humanize-duration'

const SETTINGS = ['imagenet.enabled', 'landmarks.enabled', 'faces.enabled', 'musicnn.enabled', 'movinet.enabled', 'ffmpeg_binary', 'faces.status', 'imagenet.status', 'landmarks.status', 'movinet.status', 'musicnn.status', 'faces.lastFile', 'imagenet.lastFile', 'landmarks.lastFile', 'movinet.lastFile', 'musicnn.lastFile', 'faces.batchSize', 'imagenet.batchSize', 'landmarks.batchSize', 'movinet.batchSize', 'musicnn.batchSize', 'clusterFaces.status', 'clusterFaces.lastRun']

const BOOLEAN_SETTINGS = ['imagenet.enabled', 'landmarks.enabled', 'faces.enabled', 'musicnn.enabled', 'movinet.enabled', 'faces.status', 'imagenet.status', 'landmarks.status', 'movinet.status', 'musicnn.status', 'faces.lastFile', 'imagenet.lastFile', 'landmarks.lastFile', 'movinet.lastFile', 'musicnn.lastFile', 'clusterFaces.status']

const MAX_RELATIVE_DATE = 1000 * 60 * 60 * 24 * 7 // one week

export default {
	name: 'ViewAdmin',
	components: { NcSettingsSection, NcNoteCard, NcCheckboxRadioSwitch, NcTextField },

	data() {
		return {
			loading: false,
			success: false,
			error: '',
			count: -1,
			countQueued: null,
			settings: SETTINGS.reduce((obj, key) => ({ ...obj, [key]: '' }), {}),
			timeout: null,
			daemon: undefined,
			ffmpeg: undefined,
			cron: undefined,
			modelsDownloaded: null,
			imagenetJobs: null,
			facesJobs: null,
			landmarksJobs: null,
			movinetJobs: null,
			musicnnJobs: null,
			clusterFacesJobs: null,
			tagsEnabled: null,
		}
	},

	computed: {
	},

	watch: {
		error(error) {
			if (!error) return
			OC.Notification.showTemporary(error)
		},
	},
	async created() {
		this.modelsDownloaded = loadState('recognize', 'modelsDownloaded')
		this.tagsEnabled = loadState('recognize', 'tagsEnabled')
		this.getCount()
		this.getDaemonStatus()
		this.getFfmpegStatus()
		this.getCronStatus()
		this.getJobsStatus('imagenet')
		this.getJobsStatus('faces')
		this.getJobsStatus('landmarks')
		this.getJobsStatus('movinet')
		this.getJobsStatus('musicnn')
		this.getJobsStatus('clusterFaces')

		setInterval(async () => {
			this.getCount()
			this.loadValue('imagenet.status')
			this.loadValue('faces.status')
			this.loadValue('landmarks.status')
			this.loadValue('movinet.status')
			this.loadValue('musicnn.status')
			this.loadValue('clusterFaces.status')
			this.getJobsStatus('imagenet')
			this.getJobsStatus('faces')
			this.getJobsStatus('landmarks')
			this.getJobsStatus('movinet')
			this.getJobsStatus('musicnn')
			this.getJobsStatus('clusterFaces')
		}, 5 * 60 * 1000)

		try {
			const settings = loadState('recognize', 'settings')
			for (const setting of SETTINGS) {
				this.settings[setting] = settings[setting]
				if (BOOLEAN_SETTINGS.includes(setting)) {
					this.settings[setting] = JSON.parse(this.settings[setting])
				}
				if (setting.includes('batchSize') && (this.settings[setting] === '' || settings[setting] === 'null')) {
					this.settings[setting] = 100
				}
			}
		} catch (e) {
			this.error = this.t('recognize', 'Failed to load settings')
			throw e
		}
	},

	methods: {
		async onReset() {
			this.loading = true
			await axios.get(generateUrl('/apps/recognize/admin/reset'))
			await this.getCount()
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},
		async onClearJobs() {
			this.loading = true
			await axios.get(generateUrl('/apps/recognize/admin/clearJobs'))
			await this.getCount()
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},
		async onResetFaces() {
			this.loading = true
			await axios.get(generateUrl('/apps/recognize/admin/resetFaces'))
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},
		async onRescan() {
			this.loading = true
			await axios.get(generateUrl('/apps/recognize/admin/recrawl'))
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},
		async getDaemonStatus() {
			const resp = await axios.get(generateUrl('/apps/recognize/admin/daemon'))
			this.daemon = resp.data
		},
		async getCount() {
			const resp = await axios.get(generateUrl('/apps/recognize/admin/countQueued'))
			this.countQueued = resp.data
		},
		async getFfmpegStatus() {
			const resp = await axios.get(generateUrl('/apps/recognize/admin/ffmpeg'))
			const { ffmpeg } = resp.data
			this.ffmpeg = ffmpeg
		},
		async getCronStatus() {
			const resp = await axios.get(generateUrl('/apps/recognize/admin/cron'))
			const { cron } = resp.data
			this.cron = cron
		},
		async getJobsStatus(task) {
			const resp = await axios.get(generateUrl(`/apps/recognize/admin/jobs/${task}`))
			const { scheduled, lastRun } = resp.data
			this[task + 'Jobs'] = { scheduled, lastRun }
		},
		onChange() {
			if (this.timeout) {
				clearTimeout(this.timeout)
			}
			setTimeout(() => {
				this.submit()
			}, 1000)
		},

		async submit() {
			this.loading = true
			for (const setting in this.settings) {
				if (setting.includes('status') || setting.includes('lastFile')) {
					continue
				}
				await this.setValue(setting, this.settings[setting])
			}
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},

		async loadValue(setting) {
			this.settings[setting] = await this.getValue(setting)
			if (BOOLEAN_SETTINGS.includes(setting)) {
				this.settings[setting] = JSON.parse(this.settings[setting])
			}
		},
		async setValue(setting, value) {
			try {
				if (BOOLEAN_SETTINGS.includes(setting)) {
					value = JSON.stringify(value)
				}
				await axios.put(generateUrl(`/apps/recognize/admin/settings/${setting}`), {
					value,
				})
			} catch (e) {
				this.error = this.t('recognize', 'Failed to save settings')
				throw e
			}
		},

		async getValue(setting) {
			try {
				const res = await axios.get(generateUrl(`/apps/recognize/admin/settings/${setting}`))
				if (res.status !== 200) {
					this.error = this.t('recognize', 'Failed to load settings')
					console.error('Failed request', res)
					return
				}
				return res.data.value
			} catch (e) {
				this.error = this.t('recognize', 'Failed to load settings')
				throw e
			}
		},

		showDate(timestamp) {
			if (!timestamp) {
				return this.t('recognize', 'never')
			}
			const date = new Date(Number(timestamp) * 1000)
			const age = Date.now() - date
			if (age < MAX_RELATIVE_DATE) {
				const duration = humanizeDuration(age, {
					language: OC.getLanguage().split('-')[0],
					units: ['d', 'h', 'm', 's'],
					largest: 1,
					round: true,
				})
				return this.t('recognize', '{time} ago', { time: duration })
			} else {
				return date.toLocaleDateString()
			}
		},
	},
}
</script>
<style>
figure[class^='icon-'] {
	display: inline-block;
}

#recognize {
	position: relative;
}

#recognize .loading,
#recognize .success {
	position: fixed;
	top: 70px;
	right: 20px;
}

#recognize a:link, #recognize a:visited, #recognize a:hover {
	text-decoration: underline;
}
</style>
